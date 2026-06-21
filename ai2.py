import torch 
import torch.nn as nn
import torch.optim as optim

txt=open("t.txt",'r')
text=txt.read().lower()
char = text.split()
vocab = sorted(list(set(char)))

stoi = {w:i for i,w in enumerate(vocab)}
itos = {i:w for i,w in enumerate(vocab)}

encode = lambda s: [stoi.get(w,0) for w in s.split()]
decode = lambda l: ' '.join([itos[i] for i in l])
date = [stoi.get(w,0) for w in char]
date = torch.tensor(date, dtype=torch.long)


x = date[:-1]
y = date[1:]  
vocbe_size = max(len(vocab),y.max().item()+1)
model=nn.Sequential(
    nn.Embedding(vocbe_size,64),
    nn.Linear(64,64),
    nn.ReLU(),
    nn.Linear(64,vocbe_size)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
for epoch in range(5000):
    optimizer.zero_grad()
    all_output = model(x)
    loss = criterion(all_output, y)
    loss.backward()
    optimizer.step()
    print(f"loss:{loss.item()}         epoch:{epoch}       epoch(%){epoch/50} %")

while True:
    test_input = input("you input:")
    token = torch.tensor([stoi.get(w,0) for w in test_input.split()], dtype=torch.long  )

    for _ in range(2):
        test_tensor = torch.tensor(token, dtype=torch.long)
        with torch.no_grad():
            output = model(test_tensor)
            #predicted_index = torch.argmax(output[-1]).item()
            temperature = 0.1
            logits = output[-1] / temperature
            probs = torch.softmax(logits, dim=-1)
            predicted_index = torch.multinomial(probs, num_samples=1).item()
            predicted_char = itos.get(predicted_index, '')
            token = torch.cat([token, torch.tensor([predicted_index])])
            
        print(" ".join([itos.get(i, '') for i in token.tolist()]))
