def MLP(in_features=1, hidden_features=20, out_features=1):
    hidden = nn.Linear(in_features =hidden_features, out_features=out_features, bias=True)
    activation = nn.ReLU() ##ReLU 활성화함수 :모델 정의할 때 지정
    output = nn.Linear(in_features=hidden_features, out_features=out_features, bias=True)
    net = nn.Sequential(hidden, activation, output)
    return net

##복잡한 모델은 함수보다 nn.Module()을 상속받아 정의

