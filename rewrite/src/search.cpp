#include "search.h"
#include <algorithm>
#include <cmath>

namespace leviathan {

void SearchEngine::clear(){
    tt_.clear();
    quiet_history_ = {};
    killers_ = {};
}

bool SearchEngine::time_up(){
    if(!use_deadline_) return false;
    if((nodes_ & 255ULL) != 0) return false;
    if(std::chrono::steady_clock::now() >= deadline_){ stopped_=true; return true; }
    return false;
}

bool SearchEngine::repeated(uint64_t key) const {
    int count=0;
    for(uint64_t k:history_) if(k==key && ++count>=3) return true;
    return false;
}

bool SearchEngine::has_legal_move(Position& p) const {
    const Color us=p.side_to_move();
    for(Move m:p.pseudo_legal_moves(false)){
        UndoState undo;
        if(!p.make_move(m,undo)) continue;
        const bool legal=!p.in_check(us);
        p.unmake_move(m,undo);
        if(legal) return true;
    }
    return false;
}

bool SearchEngine::history_sensitive(const Position& p) const {
    if(p.halfmove_clock() >= 80) return true;
    const size_t horizon = std::min<size_t>(history_.size(), static_cast<size_t>(std::max(0, p.halfmove_clock()) + 1));
    const size_t begin = history_.size() - horizon;
    const uint64_t key = p.key();
    int seen=0;
    for(size_t i=begin;i<history_.size();++i)
        if(history_[i]==key && ++seen>=2) return true;
    return false;
}

uint64_t SearchEngine::context_key(const Position& p) const {
    const uint64_t base = p.key();
    if(!history_sensitive(p)) return base;

    uint64_t h = base ^ 0x9E3779B97F4A7C15ULL ^ (uint64_t(p.halfmove_clock()) * 0xD6E8FEB86659FD93ULL);
    const size_t horizon = std::min<size_t>(history_.size(), static_cast<size_t>(std::max(0, p.halfmove_clock()) + 1));
    const size_t begin = history_.size() - horizon;
    for(size_t i=begin;i<history_.size();++i){
        uint64_t x=history_[i]+0x9E3779B97F4A7C15ULL+(uint64_t(i-begin)<<1);
        x^=x>>30; x*=0xBF58476D1CE4E5B9ULL;
        x^=x>>27; x*=0x94D049BB133111EBULL;
        x^=x>>31;
        h^=x+(h<<6)+(h>>2);
    }
    return h;
}

int SearchEngine::score_to_tt(int score,int ply){
    if(score > MATE-1000) return score+ply;
    if(score < -MATE+1000) return score-ply;
    return score;
}

int SearchEngine::score_from_tt(int score,int ply){
    if(score > MATE-1000) return score-ply;
    if(score < -MATE+1000) return score+ply;
    return score;
}

void SearchEngine::reward_quiet(Move m,int depth,int ply){
    int& h=quiet_history_[m.from][m.to];
    const int bonus=std::min(2048, 32*depth*depth);
    h += bonus - h*bonus/16384;
    h=std::clamp(h,-16384,16384);
    if(ply<MAX_PLY){
        if(!(killers_[ply][0]==m)){
            killers_[ply][1]=killers_[ply][0];
            killers_[ply][0]=m;
        }
    }
}

MoveList SearchEngine::ordered_moves(const Position& p,Move tt_move,int ply,bool captures_only) const {
    MoveList moves=p.pseudo_legal_moves(captures_only);
    auto move_score=[&](Move m){
        if(m==tt_move && !tt_move.is_null()) return 1000000;
        int s=0;
        if(m.flags&Move::Capture){
            Piece victim=p.piece_at(m.to);
            Piece attacker=p.piece_at(m.from);
            if(m.flags&Move::EnPassant) victim=make_piece(opposite(p.side_to_move()),PieceType::Pawn);
            s += 100000 + piece_value(type_of(victim))*16 - piece_value(type_of(attacker));
        }
        if(m.promotion) s += 90000 + piece_value(static_cast<PieceType>(m.promotion));
        if(!(m.flags&Move::Capture) && !m.promotion){
            if(ply<MAX_PLY && m==killers_[ply][0]) s+=80000;
            else if(ply<MAX_PLY && m==killers_[ply][1]) s+=70000;
            s+=quiet_history_[m.from][m.to];
        }
        return s;
    };
    std::sort(moves.begin(),moves.end(),[&](Move a,Move b){return move_score(a)>move_score(b);});
    return moves;
}

int SearchEngine::quiescence(Position& p,int alpha,int beta,int ply){
    ++nodes_; if(time_up()) return alpha;
    const uint64_t boardKey=p.key();
    const Color us=p.side_to_move();
    const bool checked=p.in_check(us);
    if(repeated(boardKey)) return 0;
    if(p.halfmove_clock()>=100){
        // Checkmate ends the game before a 50-move draw can be claimed.
        if(checked && !has_legal_move(p)) return -MATE+ply;
        return 0;
    }

    int stand=checked ? -INF : evaluator_->evaluate(p).mean_cp;
    if(!checked){
        if(stand>=beta) return beta;
        if(stand>alpha) alpha=stand;
    }

    int legalCount=0;
    for(Move m:ordered_moves(p,{},ply,!checked)){
        UndoState undo;
        if(!p.make_move(m,undo)) continue;
        if(p.in_check(us)){
            p.unmake_move(m,undo);
            continue;
        }
        ++legalCount;
        history_.push_back(p.key());
        int score=-quiescence(p,-beta,-alpha,ply+1);
        history_.pop_back();
        p.unmake_move(m,undo);
        if(stopped_) return alpha;
        if(score>=beta) return beta;
        if(score>alpha) alpha=score;
    }
    if(checked && legalCount==0) return -MATE+ply;
    return alpha;
}

int SearchEngine::negamax(Position& p,int depth,int alpha,int beta,int ply){
    ++nodes_; if(time_up()) return alpha;
    const uint64_t boardKey=p.key();
    const Color us=p.side_to_move();
    const bool inCheck=p.in_check(us);
    if(repeated(boardKey)) return 0;
    if(p.halfmove_clock()>=100){
        if(inCheck && !has_legal_move(p)) return -MATE+ply;
        return 0;
    }
    if(ply>=MAX_PLY-1) return evaluator_->evaluate(p).mean_cp;
    const uint64_t key=context_key(p);
    if(depth<=0) return quiescence(p,alpha,beta,ply);

    const int originalAlpha=alpha;
    Move ttMove{};
    if(const TTEntry* e=tt_.probe(key)){
        ++tt_hits_;
        ttMove=e->best;
        const int ttScore=score_from_tt(e->score,ply);
        if(e->depth>=depth){
            if(e->bound==Bound::Exact) return ttScore;
            if(e->bound==Bound::Lower && ttScore>=beta) return ttScore;
            if(e->bound==Bound::Upper && ttScore<=alpha) return ttScore;
        }
    }

    MoveList moves=ordered_moves(p,ttMove,ply,false);

    int bestScore=-INF;
    Move best{};
    int legalCount=0;
    size_t searchedIndex=0;
    for(Move m:moves){
        UndoState undo;
        if(!p.make_move(m,undo)) continue;
        if(p.in_check(us)){
            p.unmake_move(m,undo);
            continue;
        }
        ++legalCount;
        const bool quiet=!(m.flags&Move::Capture) && !m.promotion;
        int reduction=0;
        if(searchedIndex>=4 && depth>=3 && quiet && !inCheck){
            reduction=1;
            if(searchedIndex>=8 && depth>=6) reduction=2;
        }

        history_.push_back(p.key());
        int score;
        if(searchedIndex==0){
            score=-negamax(p,depth-1,-beta,-alpha,ply+1);
        } else {
            const int reducedDepth=std::max(0,depth-1-reduction);
            score=-negamax(p,reducedDepth,-alpha-1,-alpha,ply+1);
            if(!stopped_ && reduction && score>alpha)
                score=-negamax(p,depth-1,-alpha-1,-alpha,ply+1);
            if(!stopped_ && score>alpha && score<beta)
                score=-negamax(p,depth-1,-beta,-alpha,ply+1);
        }
        history_.pop_back();
        p.unmake_move(m,undo);
        if(stopped_) return alpha;

        if(score>bestScore){ bestScore=score; best=m; }
        if(score>alpha) alpha=score;
        ++searchedIndex;
        if(alpha>=beta){
            if(quiet) reward_quiet(m,depth,ply);
            break;
        }
    }

    if(legalCount==0) return inCheck ? -MATE+ply : 0;

    Bound b=Bound::Exact;
    if(bestScore<=originalAlpha) b=Bound::Upper;
    else if(bestScore>=beta) b=Bound::Lower;
    TTEntry out{};
    out.key=key;
    out.score=score_to_tt(bestScore,ply);
    out.depth=static_cast<int16_t>(depth);
    out.bound=b;
    out.best=best;
    tt_.store(out);
    ++tt_stores_;
    return bestScore;
}

std::vector<Move> SearchEngine::extract_pv(Position p,int max_len,std::vector<uint64_t> history) const {
    std::vector<Move> pv;
    auto contextual=[&](const Position& pos){
        const uint64_t base=pos.key();
        const size_t horizon=std::min<size_t>(history.size(), static_cast<size_t>(std::max(0,pos.halfmove_clock())+1));
        const size_t begin=history.size()-horizon;
        int seen=0;
        for(size_t j=begin;j<history.size();++j) if(history[j]==base) ++seen;
        const bool sensitive=pos.halfmove_clock()>=80 || seen>=2;
        if(!sensitive) return base;
        uint64_t h=base^0x9E3779B97F4A7C15ULL^(uint64_t(pos.halfmove_clock())*0xD6E8FEB86659FD93ULL);
        for(size_t j=begin;j<history.size();++j){
            uint64_t x=history[j]+0x9E3779B97F4A7C15ULL+(uint64_t(j-begin)<<1);
            x^=x>>30; x*=0xBF58476D1CE4E5B9ULL; x^=x>>27; x*=0x94D049BB133111EBULL; x^=x>>31;
            h^=x+(h<<6)+(h>>2);
        }
        return h;
    };
    for(int i=0;i<max_len;++i){
        const TTEntry* e=tt_.probe(contextual(p));
        if(!e || e->best.is_null()) break;
        Move m=e->best;
        bool legal=false;
        for(Move x:p.legal_moves()) if(x==m){legal=true;break;}
        if(!legal) break;
        pv.push_back(m);
        p.make_move(m);
        history.push_back(p.key());
    }
    return pv;
}

SearchReport SearchEngine::search(const Position& root,const SearchLimits& limits,const std::vector<uint64_t>& game_history){
    nodes_=0;
    tt_hits_=0;
    tt_stores_=0;
    stopped_=false;
    history_=game_history;
    if(history_.empty() || history_.back()!=root.key()) history_.push_back(root.key());
    use_deadline_=limits.movetime_ms>0;
    if(use_deadline_) deadline_=std::chrono::steady_clock::now()+std::chrono::milliseconds(limits.movetime_ms);

    const int targetMaxDepth = limits.max_depth > 0
        ? std::min(limits.max_depth, MAX_PLY - 1)
        : (use_deadline_ ? MAX_PLY - 1 : 5);

    Position work=root;
    SearchReport report{};
    int previousScore=0;
    for(int depth=1;depth<=targetMaxDepth;++depth){
        int alpha=-INF, beta=INF, window=35;
        if(depth>=3){
            alpha=std::max(-INF,previousScore-window);
            beta=std::min(INF,previousScore+window);
        }

        int score=0;
        while(true){
            score=negamax(work,depth,alpha,beta,0);
            if(stopped_) break;
            if(score<=alpha && alpha>-INF){
                window=std::min(4000,window*2);
                alpha=std::max(-INF,score-window);
                beta=std::min(INF,score+window);
                continue;
            }
            if(score>=beta && beta<INF){
                window=std::min(4000,window*2);
                alpha=std::max(-INF,score-window);
                beta=std::min(INF,score+window);
                continue;
            }
            break;
        }
        if(stopped_) break;

        previousScore=score;
        if(const TTEntry* e=tt_.probe(context_key(root))) report.best=e->best;
        report.score=score;
        report.completed_depth=depth;
        report.nodes=nodes_;
        report.tt_hits=tt_hits_;
        report.tt_stores=tt_stores_;
        report.pv=extract_pv(root,depth,history_);
    }
    if(report.best.is_null()){
        auto moves=root.legal_moves();
        if(!moves.empty()) report.best=moves.front();
    }
    report.nodes=nodes_;
    report.tt_hits=tt_hits_;
    report.tt_stores=tt_stores_;
    return report;
}

} // namespace leviathan
