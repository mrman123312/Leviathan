#include "chess.h"
#include <algorithm>
#include <cctype>
#include <charconv>
#include <sstream>

namespace leviathan {

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

Position::Position() { board_.fill(Piece::Empty); }

Position Position::startpos() {
    auto p = from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    return *p;
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

std::optional<Position> Position::from_fen(std::string_view fen) {
    std::istringstream ss{std::string(fen)};
    std::string placement, stm, castle, ep;
    int half=0, full=1;
    if (!(ss >> placement >> stm >> castle >> ep)) return std::nullopt;
    ss >> half >> full;

    Position p;
    int rank=7, file=0;
    for (char c : placement) {
        if (c == '/') { if (file != 8 || rank == 0) return std::nullopt; --rank; file=0; continue; }
        if (std::isdigit(static_cast<unsigned char>(c))) {
            int n = c-'0'; if (n < 1 || n > 8 || file+n > 8) return std::nullopt; file += n; continue;
        }
        Piece pc = fen_piece(c);
        if (pc == Piece::Empty || file >= 8 || rank < 0) return std::nullopt;
        p.board_[rank*8+file] = pc; ++file;
    }
    if (rank != 0 || file != 8) return std::nullopt;
    if (stm == "w") p.side_ = Color::White; else if (stm == "b") p.side_ = Color::Black; else return std::nullopt;
    p.castling_ = 0;
    if (castle != "-") for (char c : castle) {
        if (c=='K') p.castling_|=1; else if(c=='Q') p.castling_|=2;
        else if(c=='k') p.castling_|=4; else if(c=='q') p.castling_|=8; else return std::nullopt;
    }
    p.ep_square_ = ep == "-" ? -1 : parse_square(ep);
    if (ep != "-" && p.ep_square_ < 0) return std::nullopt;
    p.halfmove_ = std::max(0, half); p.fullmove_ = std::max(1, full);
    if (p.king_square(Color::White) < 0 || p.king_square(Color::Black) < 0) return std::nullopt;
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

int Position::king_square(Color c) const {
    Piece k = make_piece(c, PieceType::King);
    for (int i=0;i<64;++i) if (board_[i]==k) return i;
    return -1;
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
    int k=king_square(c); return k>=0 && attacked(k,opposite(c));
}

static void add_promotions(std::vector<Move>& out,int from,int to,uint8_t flags){
    for(PieceType pt: {PieceType::Queen,PieceType::Rook,PieceType::Bishop,PieceType::Knight})
        out.push_back(Move{static_cast<uint8_t>(from),static_cast<uint8_t>(to),static_cast<uint8_t>(pt),flags});
}

std::vector<Move> Position::pseudo_legal_moves(bool captures_only) const {
    std::vector<Move> out; out.reserve(64);
    Color us=side_;
    for(int sq=0;sq<64;++sq){
        Piece pc=board_[sq]; if(pc==Piece::Empty||color_of(pc)!=us) continue;
        int f=sq&7,r=sq>>3; PieceType pt=type_of(pc);
        auto target_ok=[&](int to){ return board_[to]==Piece::Empty || color_of(board_[to])!=us; };
        if(pt==PieceType::Pawn){
            int dir=us==Color::White?8:-8, start=us==Color::White?1:6, promo=us==Color::White?6:1;
            int one=sq+dir;
            if(!captures_only && one>=0&&one<64&&board_[one]==Piece::Empty){
                if(r==promo) add_promotions(out,sq,one,0); else out.push_back(Move{(uint8_t)sq,(uint8_t)one,0,0});
                int two=sq+2*dir;
                if(r==start&&board_[two]==Piece::Empty) out.push_back(Move{(uint8_t)sq,(uint8_t)two,0,Move::DoublePush});
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
    for(Move m:pseudo_legal_moves(captures_only)){
        Position q=*this; Color us=side_; if(q.make_move(m)&&!q.in_check(us)) out.push_back(m);
    }
    return out;
}

bool Position::make_move(Move m) {
    if(m.from>=64||m.to>=64) return false;
    Piece pc=board_[m.from]; if(pc==Piece::Empty||color_of(pc)!=side_) return false;
    Color us=side_; Piece captured=board_[m.to];
    bool pawn=type_of(pc)==PieceType::Pawn;
    halfmove_ = (pawn || captured!=Piece::Empty || (m.flags&Move::EnPassant)) ? 0 : halfmove_+1;
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

    board_[m.to]=pc; board_[m.from]=Piece::Empty;
    if(m.flags&Move::EnPassant){ int capSq=m.to+(us==Color::White?-8:8); board_[capSq]=Piece::Empty; }
    if(m.flags&Move::Castle){
        if(m.to==6){board_[5]=board_[7];board_[7]=Piece::Empty;}
        else if(m.to==2){board_[3]=board_[0];board_[0]=Piece::Empty;}
        else if(m.to==62){board_[61]=board_[63];board_[63]=Piece::Empty;}
        else if(m.to==58){board_[59]=board_[56];board_[56]=Piece::Empty;}
    }
    if(m.promotion) board_[m.to]=make_piece(us,static_cast<PieceType>(m.promotion));
    if(m.flags&Move::DoublePush) ep_square_=m.from+(us==Color::White?8:-8);
    if(us==Color::Black) ++fullmove_;
    side_=opposite(side_);
    return true;
}

std::optional<Move> Position::parse_uci_move(std::string_view text) const {
    for(Move m:legal_moves()) if(move_to_uci(m)==text) return m;
    return std::nullopt;
}

uint64_t Position::key() const {
    uint64_t h=1469598103934665603ULL;
    auto mix=[&](uint64_t v){h^=v;h*=1099511628211ULL;};
    for(int i=0;i<64;++i) mix((uint64_t(static_cast<int>(board_[i]))<<6)|uint64_t(i));
    mix(static_cast<int>(side_)); mix(castling_); mix(uint64_t(ep_square_+1));
    return h;
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
