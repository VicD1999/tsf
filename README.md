# TO IMPLEMENT

- RNN with attention (Nearby OK: missing parallelization of the attn mechanism to speed up the training)
- Transformers (In progress)
- Initialization of the weights of the RNNs (Test)
- Teacher training
- Make a toy example to test the architectures



# TO READ

- Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation Badhannau
- Deep Transformer Models for Time Series Forecasting: The Influenza Prevalence Case
- Professor Forcing: A New Algorithm for Training Recurrent Networks, Goyal

# IDEAS

- Train an RNN for the 1st one and apply this RNN on the 2nd one and see the performances (Kind of transfer learning if we re-train on the 2nd one) 
- Use Prophet https://facebook.github.io/prophet/ to make forecasting





## Command on vega

```bash
nohup python3 main.py -c_t model/LSTM/10.model -t --rnn LSTM > file.out 2> err.log &
```



