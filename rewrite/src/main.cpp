#include "chess.h"
#include "search.h"
#include "tablebase.h"
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace leviathan;

static uint64_t perft(const Position& p,int depth){
    if(depth==0) return 1;
    uint64_t n=0;
    for(Move m:p.legal_moves()) { Position q=p; q.make_move(m); n += perft(q,depth-1); }
    return n;
}

static bool has_move(const Position& p, const char* text){
    for(Move m:p.legal_moves()) if(move_to_uci(m)==text) return true;
    return false;
}

static const char* wdl_name(Wdl wdl){
    switch(wdl){
        case Wdl::Loss: return "loss";
        case Wdl::BlessedLoss: return "blessed-loss";
        case Wdl::Draw: return "draw";
        case Wdl::CursedWin: return "cursed-win";
        case Wdl::Win: return "win";
    }
    return "unknown";
}

static bool selftest(){
    Position start=Position::startpos();
    const uint64_t expected[]={1,20,400,8902,197281};
    for(int d=1;d<=4;++d){
        auto got=perft(start,d);
        if(got!=expected[d]){std::cerr<<"selftest startpos perft failed depth "<<d<<" expected "<<expected[d]<<" got "<<got<<"\n";return false;}
    }

    auto castle=Position::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    if(!castle || !has_move(*castle,"e1g1") || !has_move(*castle,"e1c1")){
        std::cerr<<"selftest castling failed\n"; return false;
    }

    Position ep=Position::startpos();
    for(const char* mtxt: {"e2e4","a7a6","e4e5","d7d5"}){
        auto m=ep.parse_uci_move(mtxt); if(!m){std::cerr<<"selftest EP setup failed\n";return false;} ep.make_move(*m);
    }
    if(!has_move(ep,"e5d6")){std::cerr<<"selftest en-passant failed\n";return false;}

    auto promo=Position::from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1");
    if(!promo || !has_move(*promo,"a7a8q") || !has_move(*promo,"a7a8r") || !has_move(*promo,"a7a8b") || !has_move(*promo,"a7a8n")){
        std::cerr<<"selftest promotion failed\n"; return false;
    }

    Position p=Position::startpos();
    for(const char* mtxt: {"e2e4","e7e5","g1f3","b8c6","f1b5"}){
        auto m=p.parse_uci_move(mtxt); if(!m){std::cerr<<"selftest move parse failed "<<mtxt<<"\n";return false;} p.make_move(*m);
    }
    auto round=Position::from_fen(p.fen());
    if(!round || round->fen()!=p.fen()){std::cerr<<"selftest FEN roundtrip failed\n";return false;}

    // Exact C++/Python feature-contract checks from the untouched distillation holdout.
    // Expected values here are side-to-move scores; the archived JSONL stores White POV.
    struct EvalCase { const char* fen; int expected; };
    const EvalCase evalCases[] = {
        {"r3kb1r/pp1n1ppp/2p5/3PP3/2P1b1n1/N1B5/PP1NBPPP/R3K2R b KQkq - 4 14", -140},
        {"r3kb1r/pp1n1ppp/2p3b1/3Pn3/2P4P/2B1N3/PP1NBPP1/R3K2R b KQkq - 0 17", -93},
        {"r3kb1r/pp1n1pp1/2p3b1/3Pn2p/2P4P/2B1N3/PP1NBPP1/R4RK1 b kq - 1 18", -114}
    };
    for(const auto& test : evalCases){
        auto pos=Position::from_fen(test.fen);
        if(!pos){std::cerr<<"selftest distilled eval FEN parse failed\n";return false;}
        const int got=distilled_evaluator().evaluate(*pos).mean_cp;
        if(got!=test.expected){
            std::cerr<<"selftest distilled eval mismatch expected "<<test.expected<<" got "<<got<<" fen "<<test.fen<<"\n";
            return false;
        }
    }

    SearchEngine engine;
    auto report=engine.search(Position::startpos(),SearchLimits{3,0},{Position::startpos().key()});
    if(report.best.is_null() || report.completed_depth!=3 || report.nodes<100){
        std::cerr<<"selftest search failed depth="<<report.completed_depth<<" nodes="<<report.nodes<<"\n"; return false;
    }

    if(null_tablebase().probe_wdl(Position::startpos()).has_value()){
        std::cerr<<"selftest null tablebase unexpectedly returned a result\n"; return false;
    }

    std::cout<<"selftest ok\n"; return true;
}

int main(int argc,char** argv){
    if(argc>1 && std::string(argv[1])=="--selftest") return selftest()?0:1;
    Position pos=Position::startpos();
    SearchEngine search;
    FathomTablebase tablebase;
    std::vector<uint64_t> game_history{pos.key()};
    std::string line;
    while(std::getline(std::cin,line)){
        std::istringstream in(line); std::string cmd; in>>cmd;
        if(cmd=="uci"){
            std::cout<<"id name Leviathan Rewrite v2\n";
            std::cout<<"id author Leviathan Project\n";
            std::cout<<"option name SyzygyPath type string default <empty>\n";
            std::cout<<"uciok\n"<<std::flush;
        } else if(cmd=="isready"){
            std::cout<<"readyok\n"<<std::flush;
        } else if(cmd=="setoption"){
            static const std::string prefix="setoption name SyzygyPath value ";
            if(line.rfind(prefix,0)==0){
                const std::string path=line.substr(prefix.size());
                const bool ok=tablebase.initialize(path);
                std::cout<<"info string Fathom "<<(ok?"ready":"unavailable")
                         <<" max_pieces "<<tablebase.max_pieces()
                         <<" source "<<tablebase.descriptor().source_revision<<"\n"<<std::flush;
            }
        } else if(cmd=="ucinewgame"){
            search.clear(); pos=Position::startpos(); game_history={pos.key()};
        } else if(cmd=="position"){
            std::string token; in>>token;
            Position next;
            bool ok=true;
            if(token=="startpos") next=Position::startpos();
            else if(token=="fen"){
                std::string fen,part;
                for(int i=0;i<6&&in>>part;++i){ if(i) fen+=' '; fen+=part; }
                auto parsed=Position::from_fen(fen); if(!parsed) ok=false; else next=*parsed;
            } else ok=false;
            game_history.clear();
            if(ok){
                game_history.push_back(next.key());
                std::string movesWord; if(in>>movesWord && movesWord=="moves"){
                    std::string ms;
                    while(in>>ms){auto m=next.parse_uci_move(ms);if(!m){ok=false;break;}next.make_move(*m);game_history.push_back(next.key());}
                }
            }
            if(ok) pos=next; else std::cout<<"info string invalid position command\n";
        } else if(cmd=="go"){
            SearchLimits lim; lim.max_depth=5;
            std::string t; while(in>>t){ if(t=="depth") in>>lim.max_depth; else if(t=="movetime") in>>lim.movetime_ms; }
            auto r=search.search(pos,lim,game_history);
            std::cout<<"info depth "<<r.completed_depth<<" score cp "<<r.score<<" nodes "<<r.nodes<<" pv";
            for(auto m:r.pv) std::cout<<' '<<move_to_uci(m);
            std::cout<<"\nbestmove "<<(r.best.is_null()?"0000":move_to_uci(r.best))<<"\n"<<std::flush;
        } else if(cmd=="eval"){
            const auto e=default_evaluator().evaluate(pos);
            const auto& d=default_evaluator().descriptor();
            std::cout<<"info string evaluator "<<d.id<<" score "<<e.mean_cp<<" provenance "<<e.provenance<<"\n"<<std::flush;
        } else if(cmd=="perft"){
            int d=1; in>>d; std::cout<<"nodes "<<perft(pos,d)<<"\n"<<std::flush;
        } else if(cmd=="tbprobe"){
            auto result=tablebase.probe_wdl(pos);
            if(result) std::cout<<"info string tablebase wdl "<<wdl_name(*result)<<"\n";
            else std::cout<<"info string tablebase unknown\n";
            std::cout<<std::flush;
        } else if(cmd=="d"){
            std::cout<<"Fen: "<<pos.fen()<<"\n"<<std::flush;
        } else if(cmd=="quit") break;
    }
    return 0;
}
