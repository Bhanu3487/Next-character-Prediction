import streamlit as st
import torch
import torch.nn.functional as F
from torch import nn
import urllib.request
import random
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
response = urllib.request.urlopen(url)
shakespeare_text = response.read().decode("utf-8")
words = shakespeare_text.split('\n\n',800)[:-1]
random.shuffle(words)
chars = sorted(list(set('\n\n'.join(words))))
stoi = {s:i for i,s in enumerate(chars)}
itos = {i:s for s,i in stoi.items()}
block_size = 5 # context length: how many characters do we take to predict the next one?
X, Y = [], []
for w in words[:]:
  #print(w)
  context = [0] * block_size
  for ch in w + '.':
    ix = stoi[ch]
    X.append(context)
    Y.append(ix)
#     print(''.join(itos[i] for i in context), '--->', itos[ix])
    context = context[1:] + [ix] # crop and append
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X = torch.tensor(X).to(device)
Y = torch.tensor(Y).to(device)
emb_dim = 4
emb = torch.nn.Embedding(len(stoi), emb_dim)
class NextChar(nn.Module):
    def __init__(self, block_size, vocab_size, emb_dim, hidden_size, num_layers):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lins = nn.ModuleList([nn.Linear((block_size * emb_dim if i == 0 else hidden_size), hidden_size) for i in range(num_layers)])
        self.lin_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.emb(x)
        x = x.view(x.shape[0], -1)
        for lin in self.lins:
            x = torch.relu(lin(x))
        x = self.lin_out(x)
        return x
import os
os.environ["TORCH_LOGS"] = "+dynamo"
os.environ["TORCHDYNAMO_VERBOSE"] = "1"

# Suppress errors and fallback to eager execution
import torch._dynamo
torch._dynamo.config.suppress_errors = True
# Generate names from untrained model

model = NextChar(block_size, len(stoi), emb_dim, 512, 3).to(device)
model = torch.compile(model)

g = torch.Generator()
g.manual_seed(4000002)
def generate_name(model, itos, stoi, block_size, input_text = "", max_len=100):
    context = [0] * block_size
    for i in range(max_len):
        x = torch.tensor(context).view(1, -1).to(device)
        y_pred = model(x)
        ix = torch.distributions.categorical.Categorical(logits=y_pred).sample().item()
        ch = itos[ix]
#         if ch == '.':
#             break
        input_text += ch
        context = context[1:] + [ix]
    return input_text

# Create a sidebar for user inputs
st.sidebar.title('Parameters')
input_text = st.sidebar.text_input('Input Text', 'and it ended.\n')
max_len = st.sidebar.slider('Max Length', 1, 1000, 100)
num_layers = st.sidebar.slider('Number of Layers', 1, 10, 3)

# When the 'Generate' button is clicked, call the function with the user's inputs
if st.sidebar.button('Generate'):
    model = NextChar(block_size, len(stoi), emb_dim, 512, num_layers).to(device)
    model = torch.compile(model)
    g = torch.Generator()
    g.manual_seed(4000002)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    # Mini-batch training
    batch_size = 4096
    print_every = 100
    elapsed_time = []
    for epoch in range(500):
        for i in range(0, X.shape[0], batch_size):
            x = X[i:i+batch_size]
            y = Y[i:i+batch_size]
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            opt.step()
            opt.zero_grad()

    output_text = generate_name(model, itos, stoi, block_size, input_text, max_len)
    st.write(output_text)
