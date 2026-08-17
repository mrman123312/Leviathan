#include "search.h"
#include <algorithm>
#include <cmath>

namespace leviathan {

void SearchEngine::clear(){ tt_.clear(); }

bool SearchEngine::time_up(){
    if(!use_deadline_) return false;
    if((nodes_ & 2047ULL) != 0) return false;
    if(std::chrono::steady_clock::now() >= deadline_){ stopped_=true; return true; }
    return false;
}

bool SearchEngine::repeated(uint64_t key) const {
    int count=0;
    for(uint64_t k:history_) if(k==key && ++count>=3) return true;
    return false;
}

uint64_t SearchEngine::context_key(const Position& p) const {
    // A TT entry is a claim about a position under a reversible trajectory, not
    // merely a board snapshot. Mix the recent path into a separate proof identity.
    uint64_t h = p.key() ^ 0x9E3779B97F4A7C15ULL ^ (uint64_t(p.halfmove_clock()) * 0xD6E8FEB86659FD93ULL);
    const size_t horizon = std::min<size_t>(history_.size(), static_cast<size_t>(std::max(0, p.halfmove_clock()) + 1));
    const size_t begin = history_.size() - horizon;
    for(size_t i=begin;i<history_.size();++i){
        uint64_t x=history_[i] + 0x9E3779B97F4A7C15ULL + (uint64_t(i-begin)<<1);
        x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL;
        x ^= x >> 27; x *= 0x94D049BB133111EBULL;
        x ^= x >> 31;
        h ^= x + (h<<6) + (h>>2);
    }
    return h;
}

std::vector<Move> SearchEngine::ordered_moves(const Position& p, Move tt_move, bool captures_only) const {
    auto moves=p.legal_moves(captures_only);
    auto score=[&](Move m){
        if(m==tt_move && !tt_move.is_null()) return 1000000;
        int s=0;
        if(m.flags&Move::Capture){
            Piece victim=p.piece_at(m.to);
            Piece attacker=p.piece_at(m.from);
            if(m.flags&Move::EnPassant) victim=make_piece(opposite(p.side_to_move()),PieceType::Pawn);
            s += 10000 + piece_value(type_of(victim))*10 - piece_value(type_of(attacker));
        }
        if(m.promotion) s += 8000 + piece_value(static_cast<PieceType>(m.promotion));
        return s;
    };
    std::stable_sort(moves.begin(),moves.end(),[&](Move a,Move b){return score(a)>score(b);});
    return moves;
}

int SearchEngine::quiescence(const Position& p,int alpha,int beta,int ply){
    ++nodes_; if(time_up()) return alpha;
    if(p.halfmove_clock()>=100 || repeated(p.key())) return 0;
    bool checked=p.in_check(p.side_to_move());
    int stand=checked ? -INF : evaluator_->evaluate(p).mean_cp;
    if(checked){
        auto evasions=ordered_moves(p,{},false);
        if(evasions.empty()) return -MATE+ply;
        for(Move m:evasions){
            Position q=p; q.make_move(m); history_.push_back(q.key());
            int score=-quiescence(q,-beta,-alpha,ply+1);
            history_.pop_back();
            if(stopped_) return alpha;
            if(score>=beta) return beta;
            if(score>alpha) alpha=score;
        }
        return alpha;
    }
    if(stand>=beta) return beta;
    if(stand>alpha) alpha=stand;
    for(Move m:ordered_moves(p,{},true)){
        Position q=p; q.make_move(m); history_.push_back(q.key());
        int score=-quiescence(q,-beta,-alpha,ply+1);
        history_.pop_back();
        if(stopped_) return alpha;
        if(score>=beta) return beta;
        if(score>alpha) alpha=score;
    }
    return alpha;
}

int SearchEngine::negamax(const Position& p,int depth,int alpha,int beta,int ply){
    ++nodes_; if(time_up()) return alpha;
    uint64_t boardKey=p.key();
    if(p.halfmove_clock()>=100 || repeated(boardKey)) return 0;
    uint64_t key=context_key(p);
    if(depth<=0) return quiescence(p,alpha,beta,ply);

    int original_alpha=alpha;
    Move ttMove{};
    auto it=tt_.find(key);
    if(it!=tt_.end() && it->second.key==key){
        const TTEntry& e=it->second; ttMove=e.best;
        if(e.depth>=depth){
            if(e.bound==Bound::Exact) return e.score;
            if(e.bound==Bound::Lower && e.score>=beta) return e.score;
            if(e.bound==Bound::Upper && e.score<=alpha) return e.score;
        }
    }

    auto moves=ordered_moves(p,ttMove,false);
    if(moves.empty()) return p.in_check(p.side_to_move()) ? -MATE+ply : 0;

    int bestScore=-INF; Move best{};
    for(Move m:moves){
        Position q=p; q.make_move(m); history_.push_back(q.key());
        int score=-negamax(q,depth-1,-beta,-alpha,ply+1);
        history_.pop_back();
        if(stopped_) return alpha;
        if(score>bestScore){bestScore=score;best=m;}
        if(score>alpha) alpha=score;
        if(alpha>=beta) break;
    }

    Bound b=Bound::Exact;
    if(bestScore<=original_alpha) b=Bound::Upper;
    else if(bestScore>=beta) b=Bound::Lower;
    tt_[key]=TTEntry{key,depth,bestScore,b,best,0,0};
    return bestScore;
}

std::vector<Move> SearchEngine::extract_pv(Position p,int max_len,std::vector<uint64_t> history) const {
    std::vector<Move> pv;
    auto contextual=[&](const Position& pos){
        uint64_t h=pos.key() ^ 0x9E3779B97F4A7C15ULL ^ (uint64_t(pos.halfmove_clock()) * 0xD6E8FEB86659FD93ULL);
        const size_t horizon=std::min<size_t>(history.size(), static_cast<size_t>(std::max(0,pos.halfmove_clock())+1));
        const size_t begin=history.size()-horizon;
        for(size_t j=begin;j<history.size();++j){
            uint64_t x=history[j]+0x9E3779B97F4A7C15ULL+(uint64_t(j-begin)<<1);
            x^=x>>30; x*=0xBF58476D1CE4E5B9ULL; x^=x>>27; x*=0x94D049BB133111EBULL; x^=x>>31;
            h^=x+(h<<6)+(h>>2);
        }
        return h;
    };
    for(int i=0;i<max_len;++i){
        auto it=tt_.find(contextual(p)); if(it==tt_.end()||it->second.best.is_null()) break;
        Move m=it->second.best;
        bool legal=false; for(Move x:p.legal_moves()) if(x==m){legal=true;break;}
        if(!legal) break;
        pv.push_back(m); p.make_move(m); history.push_back(p.key());
    }
    return pv;
}

SearchReport SearchEngine::search(const Position& root,const SearchLimits& limits,const std::vector<uint64_t>& game_history){
    nodes_=0; stopped_=false; history_=game_history; if(history_.empty() || history_.back()!=root.key()) history_.push_back(root.key());
    use_deadline_=limits.movetime_ms>0;
    if(use_deadline_) deadline_=std::chrono::steady_clock::now()+std::chrono::milliseconds(limits.movetime_ms);
    SearchReport report{};
    for(int depth=1;depth<=std::max(1,limits.max_depth);++depth){
        int score=negamax(root,depth,-INF,INF,0);
        if(stopped_) break;
        auto it=tt_.find(context_key(root));
        if(it!=tt_.end()) report.best=it->second.best;
        report.score=score; report.completed_depth=depth; report.nodes=nodes_; report.pv=extract_pv(root,depth,history_);
    }
    if(report.best.is_null()){
        auto moves=root.legal_moves(); if(!moves.empty()) report.best=moves.front();
    }
    report.nodes=nodes_;
    return report;
}

} // namespace leviathan
