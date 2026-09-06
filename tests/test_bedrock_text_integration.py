import importlib.util
import unittest
HAS_TORCH=importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from test_bedrock_neural import TinyLM
    from leviathan.bedrock.neural import FrozenExecutor,FrozenPolicy
    from leviathan.bedrock.runtime import BedrockRuntime,Pulse

    class Tokenizer:
        def __call__(self,text,**kwargs):return {"input_ids":torch.tensor([[1,2,3,4]])}
        def decode(self,ids,**kwargs):return " ".join(map(str,ids))

@unittest.skipUnless(HAS_TORCH,"PyTorch optional")
class BedrockTextIntegrationTests(unittest.TestCase):
    def test_one_neural_model_through_existing_cognitive_memory_graph(self):
        torch.manual_seed(44);torch.set_num_threads(1)
        model=TinyLM();engine=FrozenExecutor(model,model_id="one",revision="fixed")
        runtime=BedrockRuntime(model_id="one",neural=engine)
        result=runtime.respond("test",Tokenizer(),policies=(FrozenPolicy(),FrozenPolicy(passes=2,gain=.1)),
            max_new_tokens=3,pulses=(Pulse("hypothesis","candidate relation"),))
        self.assertTrue(result["graph_complete"])
        self.assertEqual(result["verification_status"],"unknown")
        self.assertEqual(result["search"]["selection"],"donor_fallback")
        self.assertTrue(engine.unchanged());self.assertEqual(result["training_steps"],0)
        self.assertEqual(len(runtime.memory.records),1)
        self.assertFalse(runtime.memory.records[0].verified)
        self.assertEqual(runtime.model_id,engine.model_id)

if __name__=="__main__":unittest.main()
