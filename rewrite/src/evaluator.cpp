#include "evaluator.h"
#include "distilled_eval_weights.h"
#include <algorithm>
#include <array>
#include <cstdlib>

namespace leviathan {
namespace {

struct FeatureSummary {
    int baseline_white = 0;
    int non_pawn_material = 0;
    int psqt_correction = 0;
    std::array<std::array<int,7>,2> piece_counts{};
    std::array<std::array<int,8>,2> pawn_files{};
    std::array<std::array<int,8>,2> pawn_squares{};
    std::array<int,2> pawn_count{};
    std::array<int,2> king_square{{-1,-1}};
};

FeatureSummary summarize(const Position& p) {
    FeatureSummary s;
    for(int sq=0;sq<64;++sq){
        const Piece piece=p.piece_at(sq);
        if(piece==Piece::Empty) continue;
        const Color c=color_of(piece);
        const int ci=static_cast<int>(c);
        const PieceType pt=type_of(piece);
        const int pti=static_cast<int>(pt);
        ++s.piece_counts[ci][pti];

        int value=piece_value(pt);
        if(pt!=PieceType::Pawn && pt!=PieceType::King) s.non_pawn_material+=value;
        const int f=sq&7, r=sq>>3;
        const int center=6-(std::abs(f-3)+std::abs(r-3));
        if(pt==PieceType::Knight || pt==PieceType::Bishop) value+=center*2;
        s.baseline_white += c==Color::White ? value : -value;

        const int canonicalSq=c==Color::White ? sq : (sq^56);
        const int idx=(pti-1)*64+canonicalSq;
        s.psqt_correction += (c==Color::White ? 1 : -1) * distilled_eval::kPsqt[idx];

        if(pt==PieceType::Pawn){
            ++s.pawn_files[ci][f];
            if(s.pawn_count[ci]<8) s.pawn_squares[ci][s.pawn_count[ci]++]=sq;
        } else if(pt==PieceType::King){
            s.king_square[ci]=sq;
        }
    }
    return s;
}

int doubled_pawns(const FeatureSummary& s, Color c) {
    int total=0;
    for(int n:s.pawn_files[static_cast<int>(c)]) total+=std::max(0,n-1);
    return total;
}

int isolated_pawns(const FeatureSummary& s, Color c) {
    const auto& files=s.pawn_files[static_cast<int>(c)];
    int total=0;
    for(int f=0;f<8;++f){
        if(!files[f]) continue;
        const bool left=f>0 && files[f-1]!=0;
        const bool right=f<7 && files[f+1]!=0;
        if(!left && !right) total+=files[f];
    }
    return total;
}

int passed_pawns(const FeatureSummary& s, Color c) {
    const int ci=static_cast<int>(c);
    const int ei=static_cast<int>(opposite(c));
    int total=0;
    for(int i=0;i<s.pawn_count[ci];++i){
        const int sq=s.pawn_squares[ci][i];
        const int f=sq&7, r=sq>>3;
        bool passed=true;
        for(int j=0;j<s.pawn_count[ei] && passed;++j){
            const int esq=s.pawn_squares[ei][j];
            const int ef=esq&7, er=esq>>3;
            if(std::abs(ef-f)>1) continue;
            if((c==Color::White && er>r) || (c==Color::Black && er<r)) passed=false;
        }
        if(passed) ++total;
    }
    return total;
}

int king_shield(const Position& p, const FeatureSummary& s, Color c) {
    const int kingSq=s.king_square[static_cast<int>(c)];
    if(kingSq<0) return 0;
    const Piece pawn=make_piece(c,PieceType::Pawn);
    const int kf=kingSq&7, kr=kingSq>>3;
    const int dr=c==Color::White ? 1 : -1;
    int total=0;
    for(int df=-1;df<=1;++df){
        const int f=kf+df, r=kr+dr;
        if(f>=0&&f<8&&r>=0&&r<8&&p.piece_at(r*8+f)==pawn) ++total;
    }
    return total;
}

int distilled_correction_white(const Position& p, const FeatureSummary& s) {
    int correction=s.psqt_correction;
    const int bishopPair=(s.piece_counts[0][static_cast<int>(PieceType::Bishop)]>=2 ? 1:0)
                        -(s.piece_counts[1][static_cast<int>(PieceType::Bishop)]>=2 ? 1:0);
    const int doubled=doubled_pawns(s,Color::White)-doubled_pawns(s,Color::Black);
    const int isolated=isolated_pawns(s,Color::White)-isolated_pawns(s,Color::Black);
    const int passed=passed_pawns(s,Color::White)-passed_pawns(s,Color::Black);
    const uint8_t rights=p.castling_rights();
    const int whiteRights=((rights&1)?1:0)+((rights&2)?1:0);
    const int blackRights=((rights&4)?1:0)+((rights&8)?1:0);
    const int castlingRights=whiteRights-blackRights;
    const int shield=king_shield(p,s,Color::White)-king_shield(p,s,Color::Black);
    const int tempo=p.side_to_move()==Color::White ? 1 : -1;
    const int extras[]={bishopPair,doubled,isolated,passed,castlingRights,shield,tempo};
    for(size_t i=0;i<distilled_eval::kExtra.size();++i) correction+=extras[i]*distilled_eval::kExtra[i];
    return std::clamp(correction,-distilled_eval::kMaxCorrection,distilled_eval::kMaxCorrection);
}

} // namespace

Evaluation BaselineEvaluator::evaluate(const Position& position) const {
    return evaluate_position(position);
}

const EvaluatorDescriptor& BaselineEvaluator::descriptor() const {
    static constexpr EvaluatorDescriptor d{
        "leviathan-baseline-v0","native","none","Leviathan project",EvaluatorOrigin::Native
    };
    return d;
}

Evaluation DistilledEvaluator::evaluate(const Position& position) const {
    const FeatureSummary s=summarize(position);
    const int totalWhite=s.baseline_white+distilled_correction_white(position,s);
    const int perspective=position.side_to_move()==Color::White ? totalWhite : -totalWhite;
    const uint16_t uncertainty=static_cast<uint16_t>(std::clamp(2400-s.non_pawn_material,64,2400));
    const uint16_t volatility=static_cast<uint16_t>(position.in_check(position.side_to_move()) ? 1024 : 0);
    return Evaluation{perspective,uncertainty,volatility,2};
}

const EvaluatorDescriptor& DistilledEvaluator::descriptor() const {
    static constexpr EvaluatorDescriptor d{
        "leviathan-distilled-v1","stockfish18-teacher","linear-residual-sf18-d5-seed8910-v1","project-native",EvaluatorOrigin::Hybrid
    };
    return d;
}

const Evaluator& baseline_evaluator() { static const BaselineEvaluator evaluator; return evaluator; }
const Evaluator& distilled_evaluator() { static const DistilledEvaluator evaluator; return evaluator; }
const Evaluator& default_evaluator() { return distilled_evaluator(); }

} // namespace leviathan
