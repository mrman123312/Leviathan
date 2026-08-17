#include "chess.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>

namespace leviathan {
namespace {

constexpr uint64_t mix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

constexpr uint64_t piece_key(Piece p, int sq) {
    return p == Piece::Empty ? 0ULL
        : mix64(0x4C56504B45590000ULL ^ (uint64_t(static_cast<uint8_t>(p)) << 8) ^ uint64_t(sq));
}
constexpr uint64_t side_key() { return mix64(0x4C56534944454B45ULL); }
constexpr uint64_t castling_key(uint8_t rights) {
    return mix64(0x4C56434153544C45ULL ^ uint64_t(rights));
}
constexpr uint64_t ep_key(int sq) {
    return mix64(0x4C56455053515541ULL ^ uint64_t(sq + 1));
}

static Piece fen_piece(char c) {
    switch (c) {
        case 'P': return Piece::WP; case 'N': return Piece::WN; case 'B': return Piece::WB;
        case 'R': return Piece::WR; case 'Q': return Piece::WQ; case 'K': return Piece::WK;
        case 'p': return Piece::BP; case 'n': return Piece::BN; case 'b': return Piece::BB;
        case 'r': return Piece::BR; case 'q': return Piece::BQ; case 'k': return Piece::BK;
        default: return Piece::Empty;
    }
}

static char piece_fen(Piece p) {
    static constexpr char chars[] = ".PNBRQKpnbrqk";
    return chars[static_cast<int>(p)];
}

static void add_promotions(MoveList& out,int from,int to,uint8_t flags){
    for(PieceType pt: {PieceType::Queen,PieceType::Rook,PieceType::Bishop,PieceType::Knight})
        out.push_back(Move{static_cast<uint8_t>(from),static_cast<uint8_t>(to),static_cast<uint8_t>(pt),flags});
}

} // namespace

Color color_of(Piece p) {
    return static_cast<int>(p) <= 6 ? Color::White : Color::Black;
}

PieceType type_of(Piece p) {
    if (p == Piece::Empty) return PieceType::None;
    int v = static_cast<int>(p);
    if (v > 6) v -= 6;
    return static_cast<PieceType>(v);
}

Piece make_piece(Color c, PieceType pt) {
    if (pt == PieceType::None) return Piece::Empty;
    int v = static_cast<int>(pt) + (c == Color::Black ? 6 : 0);
    return static_cast<Piece>(v);
}

std::string square_name(int sq) {
    if (sq < 0 || sq >= 64) return "-";
    std::string s(2, ' ');
    s[0] = static_cast<char>('a' + (sq & 7));
    s[1] = static_cast<char>('1' + (sq >> 3));
    return s;
}

int parse_square(std::string_view s) {
    if (s.size() != 2 || s[0] < 'a' || s[0] > 'h' || s[1] < '1' || s[1] > '8') return -1;
    return (s[1]-'1')*8 + (s[0]-'a');
}

std::string move_to_uci(Move m) {
    std::string out = square_name(m.from) + square_name(m.to);
    if (m.promotion) {
        char c = 'q';
        switch (static_cast<PieceType>(m.promotion)) {
            case PieceType::Knight: c='n'; break;
            case PieceType::Bishop: c='b'; break;
            case PieceType::Rook: c='r'; break;
            case PieceType::Queen: c='q'; break;
            default: break;
        }
        out.push_back(c);
    }
    return out;
}

Position::Position() {
    board_.fill(Piece::Empty);
    king_sq_.fill(-1);
    recompute_key();
}

int Position::canonical_ep_square() const {
    if(ep_square_ < 0 || ep_square_ >= 64) return -1;
    if(board_[ep_square_] != Piece::Empty) return -1;

    const Color us = side_;
    const int dir = us == Color::White ? 8 : -8;
    const int capturedSq = ep_square_ - dir;
    if(capturedSq < 0 || capturedSq >= 64) return -1;
    if(board_[capturedSq] != make_piece(opposite(us), PieceType::Pawn)) return -1;

    const Piece ownPawn = make_piece(us, PieceType::Pawn);
    const int targetFile = ep_square_ & 7;
    for(int df : {-1, 1}) {
        const int from = ep_square_ - dir + df;
        if(from < 0 || from >= 64) continue;
        if(std::abs((from & 7) - targetFile) != 1) continue;
        if(board_[from] != ownPawn) continue;

        Position q = *this;
        q.board_[from] = Piece::Empty;
        q.board_[capturedSq] = Piece::Empty;
        q.board_[ep_square_] = ownPawn;
        if(!q.in_check(us)) return ep_square_;
    }
    return -1;
}

void Position::recompute_key() {
    uint64_t h = castling_key(castling_) ^ ep_key(canonical_ep_square());
    if(side_ == Color::Black) h ^= side_key();
    for(int sq=0;sq<64;++sq) h ^= piece_key(board_[sq],sq);
    key_ = h;
}

void Position::set_piece(int sq, Piece p) {
    const Piece old = board_[sq];
    if(old == p) return;
    key_ ^= piece_key(old,sq);
    board_[sq] = p;
    key_ ^= piece_key(p,sq);
}

Position Position::startpos() {
    auto p = from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    return *p;
}

std::optional<Position> Position::from_fen(std::string_view fen) {
    std::istringstream ss{std::string(fen)};
    std::string placement, stm, castle, ep;
    int half=0, full=1;
    if (!(ss >> placement >> stm >> castle >> ep)) return std::nullopt;
    ss >> half >> full;

    Position p;
    p.board_.fill(Piece::Empty);
    p.king_sq_.fill(-1);
    int whiteKings=0, blackKings=0;
    int rank=7, file=0;
    for (char c : placement) {
        if (c == '/') { if (file != 8 || rank == 0) return std::nullopt; --rank; file=0; continue; }
        if (std::isdigit(static_cast<unsigned char>(c))) {
            int n = c-'0'; if (n < 1 || n > 8 || file+n > 8) return std::nullopt; file += n; continue;
        }
        Piece pc = fen_piece(c);
        if (pc == Piece::Empty || file >= 8 || rank < 0) return std::nullopt;
        const int sq=rank*8+file;
        p.board_[sq] = pc;
        if(pc==Piece::WK){p.king_sq_[0]=static_cast<int8_t>(sq);++whiteKings;}
        if(pc==Piece::BK){p.king_sq_[1]=static_cast<int8_t>(sq);++blackKings;}
        ++file;
    }
    if (rank != 0 || file != 8 || whiteKings!=1 || blackKings!=1) return std::nullopt;
    if (stm == "w") p.side_ = Color::White; else if (stm == "b") p.side_ = Color::Black; else return std::nullopt;
    p.castling_ = 0;
    if (castle != "-") for (char c : castle) {
        if (c=='K') p.castling_|=1; else if(c=='Q') p.castling_|=2;
        else if(c=='k') p.castling_|=4; else if(c=='q') p.castling_|=8; else return std::nullopt;
    }
    p.ep_square_ = ep == "-" ? -1 : parse_square(ep);
    if (ep != "-" && p.ep_square_ < 0) return std::nullopt;
    p.halfmove_ = std::max(0, half);
    p.fullmove_ = std::max(1, full);
    p.recompute_key();
    return p;
}

std::string Position::fen() const {
    std::ostringstream out;
    for (int r=7; r>=0; --r) {
        int empty=0;
        for (int f=0; f<8; ++f) {
            Piece pc=board_[r*8+f];
            if (pc==Piece::Empty) ++empty;
            else { if (empty) { out << empty; empty=0; } out << piece_fen(pc); }
        }
        if (empty) out << empty;
        if (r) out << '/';
    }
    out << (side_==Color::White ? " w " : " b ");
    if (!castling_) out << '-'; else {
        if (castling_&1) out << 'K';
        if (castling_&2) out << 'Q';
        if (castling_&4) out << 'k';
        if (castling_&8) out << 'q';
    }
    out << ' ' << (ep_square_>=0 ? square_name(ep_square_) : "-") << ' ' << halfmove_ << ' ' << fullmove_;
    return out.str();
}

bool Position::attacked(int sq, Color by) const {
    int f=sq&7, r=sq>>3;
    if (by==Color::White) {
        if (r>0 && f>0 && board_[sq-9]==Piece::WP) return true;
        if (r>0 && f<7 && board_[sq-7]==Piece::WP) return true;
    } else {
        if (r<7 && f>0 && board_[sq+7]==Piece::BP) return true;
        if (r<7 && f<7 && board_[sq+9]==Piece::BP) return true;
    }

    static constexpr int kN[8][2]={{1,2},{2,1},{2,-1},{1,-2},{-1,-2},{-2,-1},{-2,1},{-1,2}};
    Piece knight=make_piece(by,PieceType::Knight);
    for (auto &d:kN) { int nf=f+d[0], nr=r+d[1]; if(nf>=0&&nf<8&&nr>=0&&nr<8&&board_[nr*8+nf]==knight) return true; }

    Piece king=make_piece(by,PieceType::King);
    for(int df=-1;df<=1;++df) for(int dr=-1;dr<=1;++dr) if(df||dr){int nf=f+df,nr=r+dr;if(nf>=0&&nf<8&&nr>=0&&nr<8&&board_[nr*8+nf]==king)return true;}

    const int dirs[8][2]={{1,0},{-1,0},{0,1},{0,-1},{1,1},{1,-1},{-1,1},{-1,-1}};
    for(int i=0;i<8;++i){
        int nf=f+dirs[i][0],nr=r+dirs[i][1];
        while(nf>=0&&nf<8&&nr>=0&&nr<8){
            Piece pc=board_[nr*8+nf];
            if(pc!=Piece::Empty){
                if(color_of(pc)==by){
                    PieceType pt=type_of(pc);
                    if(pt==PieceType::Queen || (i<4&&pt==PieceType::Rook) || (i>=4&&pt==PieceType::Bishop)) return true;
                }
                break;
            }
            nf+=dirs[i][0];nr+=dirs[i][1];
        }
    }
    return false;
}

bool Position::in_check(Color c) const {
    const int k=king_square(c);
    return k>=0 && attacked(k,opposite(c));
}

MoveList Position::pseudo_legal_moves(bool captures_only) const {
    MoveList out;
    Color us=side_;
    for(int sq=0;sq<64;++sq){
        Piece pc=board_[sq]; if(pc==Piece::Empty||color_of(pc)!=us) continue;
        int f=sq&7,r=sq>>3; PieceType pt=type_of(pc);
        auto target_ok=[&](int to){ return board_[to]==Piece::Empty || color_of(board_[to])!=us; };
        if(pt==PieceType::Pawn){
            int dir=us==Color::White?8:-8, start=us==Color::White?1:6, promo=us==Color::White?6:1;
            int one=sq+dir;
            if(one>=0&&one<64&&board_[one]==Piece::Empty){
                if(r==promo) add_promotions(out,sq,one,0);
                else if(!captures_only){
                    out.push_back(Move{(uint8_t)sq,(uint8_t)one,0,0});
                    int two=sq+2*dir;
                    if(r==start&&board_[two]==Piece::Empty) out.push_back(Move{(uint8_t)sq,(uint8_t)two,0,Move::DoublePush});
                }
            }
            for(int df: {-1,1}){
                int nf=f+df; if(nf<0||nf>7) continue;
                int to=sq+dir+df; if(to<0||to>=64) continue;
                bool ep=(to==ep_square_);
                if((board_[to]!=Piece::Empty&&color_of(board_[to])!=us)||ep){
                    uint8_t flags=Move::Capture | (ep?Move::EnPassant:0);
                    if(r==promo) add_promotions(out,sq,to,flags); else out.push_back(Move{(uint8_t)sq,(uint8_t)to,0,flags});
                }
            }
        } else if(pt==PieceType::Knight){
            static constexpr int d[8][2]={{1,2},{2,1},{2,-1},{1,-2},{-1,-2},{-2,-1},{-2,1},{-1,2}};
            for(auto &x:d){int nf=f+x[0],nr=r+x[1];if(nf<0||nf>7||nr<0||nr>7)continue;int to=nr*8+nf;if(!target_ok(to))continue;bool cap=board_[to]!=Piece::Empty;if(captures_only&&!cap)continue;out.push_back(Move{(uint8_t)sq,(uint8_t)to,0,(uint8_t)(cap?Move::Capture:0)});}
        } else if(pt==PieceType::Bishop||pt==PieceType::Rook||pt==PieceType::Queen){
            static constexpr int d[8][2]={{1,0},{-1,0},{0,1},{0,-1},{1,1},{1,-1},{-1,1},{-1,-1}};
            int a=pt==PieceType::Bishop?4:0,b=pt==PieceType::Rook?4:8;
            for(int i=a;i<b;++i){int nf=f+d[i][0],nr=r+d[i][1];while(nf>=0&&nf<8&&nr>=0&&nr<8){int to=nr*8+nf;Piece t=board_[to];if(t==Piece::Empty){if(!captures_only)out.push_back(Move{(uint8_t)sq,(uint8_t)to,0,0});}else{if(color_of(t)!=us)out.push_back(Move{(uint8_t)sq,(uint8_t)to,0,Move::Capture});break;}nf+=d[i][0];nr+=d[i][1];}}
        } else if(pt==PieceType::King){
            for(int df=-1;df<=1;++df)for(int dr=-1;dr<=1;++dr)if(df||dr){int nf=f+df,nr=r+dr;if(nf<0||nf>7||nr<0||nr>7)continue;int to=nr*8+nf;if(!target_ok(to))continue;bool cap=board_[to]!=Piece::Empty;if(captures_only&&!cap)continue;out.push_back(Move{(uint8_t)sq,(uint8_t)to,0,(uint8_t)(cap?Move::Capture:0)});}
            if(!captures_only && !in_check(us)){
                Color them=opposite(us);
                if(us==Color::White && sq==4){
                    if((castling_&1)&&board_[5]==Piece::Empty&&board_[6]==Piece::Empty&&board_[7]==Piece::WR&&!attacked(5,them)&&!attacked(6,them)) out.push_back(Move{4,6,0,Move::Castle});
                    if((castling_&2)&&board_[3]==Piece::Empty&&board_[2]==Piece::Empty&&board_[1]==Piece::Empty&&board_[0]==Piece::WR&&!attacked(3,them)&&!attacked(2,them)) out.push_back(Move{4,2,0,Move::Castle});
                } else if(us==Color::Black && sq==60){
                    if((castling_&4)&&board_[61]==Piece::Empty&&board_[62]==Piece::Empty&&board_[63]==Piece::BR&&!attacked(61,them)&&!attacked(62,them)) out.push_back(Move{60,62,0,Move::Castle});
                    if((castling_&8)&&board_[59]==Piece::Empty&&board_[58]==Piece::Empty&&board_[57]==Piece::Empty&&board_[56]==Piece::BR&&!attacked(59,them)&&!attacked(58,them)) out.push_back(Move{60,58,0,Move::Castle});
                }
            }
        }
    }
    return out;
}

std::vector<Move> Position::legal_moves(bool captures_only) const {
    std::vector<Move> out;
    Position work=*this;
    const Color us=side_;
    for(Move m:pseudo_legal_moves(captures_only)){
        UndoState undo;
        if(!work.make_move(m,undo)) continue;
        const bool legal=!work.in_check(us);
        work.unmake_move(m,undo);
        if(legal) out.push_back(m);
    }
    return out;
}

bool Position::make_move(Move m) {
    UndoState ignored;
    return make_move(m,ignored);
}

bool Position::make_move(Move m, UndoState& undo) {
    if(m.from>=64||m.to>=64) return false;
    Piece pc=board_[m.from]; if(pc==Piece::Empty||color_of(pc)!=side_) return false;
    const Color us=side_;

    undo = UndoState{};
    undo.moved=pc;
    undo.captured_on_to=board_[m.to];
    undo.castling=castling_;
    undo.ep_square=ep_square_;
    undo.halfmove=halfmove_;
    undo.fullmove=fullmove_;
    undo.side=side_;
    undo.white_king=king_sq_[0];
    undo.black_king=king_sq_[1];
    undo.key=key_;

    const int oldCanonicalEp=canonical_ep_square();
    key_ ^= castling_key(castling_) ^ ep_key(oldCanonicalEp);

    const bool pawn=type_of(pc)==PieceType::Pawn;
    halfmove_ = (pawn || undo.captured_on_to!=Piece::Empty || (m.flags&Move::EnPassant)) ? 0 : halfmove_+1;
    ep_square_=-1;

    if(pc==Piece::WK) castling_ &= ~uint8_t(3);
    if(pc==Piece::BK) castling_ &= ~uint8_t(12);
    if(m.from==0) castling_ &= ~uint8_t(2);
    if(m.from==7) castling_ &= ~uint8_t(1);
    if(m.from==56) castling_ &= ~uint8_t(8);
    if(m.from==63) castling_ &= ~uint8_t(4);
    if(m.to==0) castling_ &= ~uint8_t(2);
    if(m.to==7) castling_ &= ~uint8_t(1);
    if(m.to==56) castling_ &= ~uint8_t(8);
    if(m.to==63) castling_ &= ~uint8_t(4);

    set_piece(m.to,pc);
    set_piece(m.from,Piece::Empty);

    if(type_of(pc)==PieceType::King)
        king_sq_[static_cast<int>(us)]=static_cast<int8_t>(m.to);

    if(m.flags&Move::EnPassant){
        const int capSq=m.to+(us==Color::White?-8:8);
        undo.ep_capture_square=capSq;
        undo.ep_captured=board_[capSq];
        set_piece(capSq,Piece::Empty);
    }
    if(m.flags&Move::Castle){
        if(m.to==6){Piece rook=board_[7];set_piece(5,rook);set_piece(7,Piece::Empty);}
        else if(m.to==2){Piece rook=board_[0];set_piece(3,rook);set_piece(0,Piece::Empty);}
        else if(m.to==62){Piece rook=board_[63];set_piece(61,rook);set_piece(63,Piece::Empty);}
        else if(m.to==58){Piece rook=board_[56];set_piece(59,rook);set_piece(56,Piece::Empty);}
    }
    if(m.promotion) set_piece(m.to,make_piece(us,static_cast<PieceType>(m.promotion)));
    if(m.flags&Move::DoublePush) ep_square_=m.from+(us==Color::White?8:-8);
    if(us==Color::Black) ++fullmove_;
    side_=opposite(side_);
    key_ ^= side_key();
    key_ ^= castling_key(castling_) ^ ep_key(canonical_ep_square());
    return true;
}

void Position::unmake_move(Move m, const UndoState& undo) {
    side_=undo.side;
    castling_=undo.castling;
    ep_square_=undo.ep_square;
    halfmove_=undo.halfmove;
    fullmove_=undo.fullmove;
    king_sq_[0]=static_cast<int8_t>(undo.white_king);
    king_sq_[1]=static_cast<int8_t>(undo.black_king);

    board_[m.from]=undo.moved;
    board_[m.to]=undo.captured_on_to;
    if(m.flags&Move::EnPassant && undo.ep_capture_square>=0)
        board_[undo.ep_capture_square]=undo.ep_captured;
    if(m.flags&Move::Castle){
        const Piece rook=make_piece(undo.side,PieceType::Rook);
        if(m.to==6){board_[7]=rook;board_[5]=Piece::Empty;}
        else if(m.to==2){board_[0]=rook;board_[3]=Piece::Empty;}
        else if(m.to==62){board_[63]=rook;board_[61]=Piece::Empty;}
        else if(m.to==58){board_[56]=rook;board_[59]=Piece::Empty;}
    }
    key_=undo.key;
}

std::optional<Move> Position::parse_uci_move(std::string_view text) const {
    for(Move m:legal_moves()) if(move_to_uci(m)==text) return m;
    return std::nullopt;
}

int piece_value(PieceType pt) {
    switch(pt){case PieceType::Pawn:return 100;case PieceType::Knight:return 320;case PieceType::Bishop:return 330;case PieceType::Rook:return 500;case PieceType::Queen:return 900;default:return 0;}
}

Evaluation evaluate_position(const Position& p) {
    int score=0;
    int nonPawnMaterial=0;
    int hangingPressure=0;
    for(int sq=0;sq<64;++sq){
        Piece pc=p.piece_at(sq); if(pc==Piece::Empty) continue;
        int v=piece_value(type_of(pc));
        if(type_of(pc)!=PieceType::Pawn && type_of(pc)!=PieceType::King) nonPawnMaterial += v;
        int f=sq&7,r=sq>>3;
        int center=6-(std::abs(f-3)+std::abs(r-3));
        if(type_of(pc)==PieceType::Knight||type_of(pc)==PieceType::Bishop) v += center*2;
        score += color_of(pc)==Color::White ? v : -v;
    }
    int perspective = p.side_to_move()==Color::White ? score : -score;
    uint16_t uncertainty = static_cast<uint16_t>(std::clamp(2400 - nonPawnMaterial, 64, 2400));
    uint16_t volatility = static_cast<uint16_t>(p.in_check(p.side_to_move()) ? 1024 : hangingPressure);
    return Evaluation{perspective, uncertainty, volatility, 1};
}

} // namespace leviathan
