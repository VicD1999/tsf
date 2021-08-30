# Test to do

- Train an RNN for the 1st one and apply this RNN on the 2nd one and see the performances (Kind of transfer learning if we re-train on the 2nd one)



# TO DO

- Train model (LSTM OK)
- Classic Encoder => Decoder RNN architecture (OK)
- RNN with attention (Nearby OK: missing parallelization of the attn mechanism to speed up the training)
- Transformers (In progress)
- Initialization of the weights of the RNNs (Test)
- Teacher training
- Make a toy example to test the architectures

## Command on vega

```bash
nohup python3 main.py -c_t model/LSTM/10.model -t --rnn LSTM > file.out 2> err.log &
```



