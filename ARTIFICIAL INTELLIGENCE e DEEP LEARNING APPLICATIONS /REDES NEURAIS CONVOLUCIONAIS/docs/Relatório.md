# Do pixel ao deploy: resolução completa dos exercícios de CNN

## Visão geral

Este relatório registra a execução dos exercícios de CNN com Fashion-MNIST como linha de base, doze ablações controladas no Nível 2, um baseline com MLP, análise de erros e filtros, e a extensão para CIFAR-10 pelo Caminho A. A implementação foi reproduzida em `cnn_junior/`; o diretório `cnn/` permanece preservado como referência do material original.

O ambiente usado foi PyTorch 2.5 em CPU, com semente 42. A linha de base atingiu 89,58% de acurácia no teste, enquanto a mesma arquitetura aplicada ao CIFAR-10 atingiu 76,18%.

| Resultado | Valor |
| --- | ---: |
| Linha de base Fashion-MNIST, teste | 89,58% |
| Treinos reais registrados | 14 |
| CIFAR-10, teste | 76,18% |
| Parâmetros da CNN de referência | 98.554 |

## Contexto da implementação

A execução foi organizada em torno de `cnn_junior/`, com `cnn/` mantido intacto como referência. Os resultados brutos das execuções ficam em `outputs/<tag>/historico.json`. A avaliação usa `evaluate.py`, e a predição de imagens individuais é feita por `predict.py`.

A arquitetura de referência é uma CNN com três blocos convolucionais, BatchNorm, ReLU e MaxPool, seguida da etapa de classificação. A mesma estrutura foi reaproveitada no Caminho A para CIFAR-10, alterando apenas a configuração do dataset.

## Metodologia

A leitura dos resultados depende de três decisões experimentais.

**Épocas por experimento.** A linha de base e o MLP (3.1) rodaram as 12 épocas completas, como no `cnn/` original. As doze ablações do Nível 2 rodaram 5 épocas cada — o que o próprio enunciado pede nos itens 2.1, 2.2 e 2.4, e o que ficou padronizado nos demais para que a comparação entre eles seja justa entre si. Isso significa que comparar um número do Nível 2 diretamente com a linha de base de 12 épocas subestima levemente o que cada configuração atingiria com treino completo — as comparações corretas são *entre os experimentos do Nível 2*, não contra a linha de base absoluta.

**Tempo por época.** Esta máquina hibernou brevemente durante a rodada (comum em notebook), e três das catorze execuções registraram uma época com duração de dezenas de minutos por causa disso — não por lentidão real de CPU. Todas as tabelas abaixo usam a **mediana** das épocas de cada execução, que ignora esse tipo de outlier; os valores brutos estão em `outputs/<tag>/historico.json` para quem quiser conferir.

**Um bug real, corrigido em público.** Na primeira rodada do experimento 2.1 (sem normalização), a função de avaliação normalizava o conjunto de teste mesmo assim — um descompasso de pré-processamento entre treino e teste, o mesmo tipo de erro que o `README` original aponta como o mais comum em deploy de visão computacional (seção 7-B). O sintoma foi uma acurácia de teste de 27,6% com validação de 88% — grande demais para ser real. O código foi corrigido (`evaluate.py` agora recebe a mesma flag `--sem-normalizacao` usada no treino) e o experimento reavaliado; o número correto, 87,68%, é o que aparece na tabela do Nível 2. O registro do ajuste faz parte dos resultados da rodada.

## Nível 1 — Compreensão

## Compreensão

### 1.1 Fórmula do tamanho

Fórmula: `saída = ⌊(entrada + 2·padding − kernel) / stride⌋ + 1`, aplicada em sequência a partir de uma entrada `(1, 1, 28, 28)`:

| Operação | Cálculo | Saída (C,H,W) |
| --- | --- | --- |
| a) `Conv2d(1,8,k=5,p=0,s=1)` | ⌊(28+0−5)/1⌋+1 = 24 | (8, 24, 24) |
| b) `MaxPool2d(2,2)` sobre (a) | 24 / 2 = 12 | (8, 12, 12) |
| c) `Conv2d(8,16,k=3,p=1,s=2)` sobre (b) | ⌊(12+2−3)/2⌋+1 = 6 | (16, 6, 6) |

Confirmado com `torch.zeros(1,1,28,28)` passado pelas camadas reais — os três formatos batem exatamente.

### 1.2 Contagem de parâmetros

| Camada | Parâmetros |
| --- | --- |
| `Conv2d(16,32,k=3)` com bias | 4.640 |
| `Conv2d(16,32,k=3)` sem bias | 4.608 |
| `Linear(16·14·14 → 32·14·14)` | 19.675.264 |

A convolução liga cada um dos 32 filtros a apenas `16×3×3` entradas (conectividade local) e reaproveita esses mesmos pesos em toda posição da imagem (pesos compartilhados) — daí 4.608–4.640 parâmetros. A `Linear` equivalente precisaria de uma conexão independente entre cada uma das 3.136 entradas e cada uma das 6.272 saídas: 19.675.264 parâmetros, **4.244× mais**. Em memória isso é a diferença entre ~18 KB e ~78,7 MB só nessa camada; em risco de overfitting, a `Linear` tem parâmetros suficientes para memorizar o dataset de treino várias vezes, enquanto a convolução é forçada a aprender um padrão local reutilizável — é literalmente menos capacidade para decorar.

### 1.3 Leitura de curvas

| Cenário | Diagnóstico | Correção |
| --- | --- | --- |
| (a) treino 99%, val 82%, perda de val subindo há 4 épocas | Overfitting clássico | Mais augmentation/dropout, weight decay; a essa altura o early stopping (paciência=3) já teria interrompido o treino |
| (b) treino 61%, val 60%, ambas as perdas estáveis e altas | Underfitting | Mais capacidade (filtros/blocos), treinar mais, ou `lr` maior |
| (c) perda de treino serrilhada, oscilando muito entre lotes | Taxa de aprendizado alta demais | Reduzir `lr`, aumentar o batch |

O cenário (c) não é hipotético aqui: o experimento 2.4 com `lr=1e-1` (Nível 2, abaixo) mostra exatamente esse padrão — a validação chegou a 52,5% no meio do treino e caiu para 33,2% na última época, evidência direta de que o passo de otimização está grande demais e "pulando" o mínimo em vez de convergir.

### 1.4 Por que sem softmax?

`nn.CrossEntropyLoss` aplica `log_softmax` internamente sobre os logits recebidos. Se o `forward` já devolvesse probabilidades (softmax aplicado), a perda calcularia `log_softmax(softmax(logits))` — um softmax sobre uma distribuição que já soma 1, o que **comprime a distribuição uma segunda vez** em vez de interpretá-la. Um exemplo numérico concreto, para logits `[5, 1, 1, 1, 1, 1, 1, 1, 1, 1]` (uma classe claramente vencedora):

|  | Prob. da classe vencedora | Distribuição completa |
| --- | --- | --- |
| 1× softmax (correto) | 85,8% | confiante, informativa |
| 2× softmax (bug) | 20,5% | quase uniforme (10% seria o acaso) |

A rede "sabia" a resposta com 85,8% de confiança, mas o segundo softmax achata isso para 20,5% — pouco acima do chute aleatório de 10%. Como o gradiente é proporcional à diferença entre a distribuição prevista e o rótulo, uma distribuição artificialmente achatada gera gradientes muito menores: o treino fica lento e, em casos extremos (logits mais espalhados que este exemplo), numericamente instável. Por isso o `forward` de `model.py` devolve logits crus, e o softmax só aparece na hora de interpretar a saída (`predict.py`, `deploy/api.py`).

## Nível 2 — Experimentação controlada

## Experimentação controlada

Cada experimento alterou uma variável em relação à configuração de referência. Foram realizados catorze treinos reais; os resultados abaixo não são estimativas.

| Experimento | Val. acc | Teste acc | Parâmetros | Tempo/época | Δ vs. base | Observação |
| --- | --- | --- | --- | --- | --- | --- |
| Linha de base | 89.93% | 89.58% | 98,554 | 52s | referência | 12 épocas · config padrão |
| 2.1 sem normalização | 88.00% | 87.68% | 98,554 | 63s | -1.9 pp | 5 épocas |
| 2.2 sem augmentation | 90.92% | 90.11% | 98,554 | 42s | +0.5 pp | 5 épocas |
| 2.3 canais (8,16,32) | 85.25% | 85.37% | 44,226 | 45s | -4.2 pp | 5 épocas |
| 2.3 canais (32,64,128) | 89.30% | 88.69% | 241,770 | 85s | -0.9 pp | 5 épocas |
| 2.4 lr = 1e-1 | 52.47% | 54.29% | 98,554 | 54s | -35.3 pp | 5 épocas |
| 2.4 lr = 1e-3 | 88.30% | 88.17% | 98,554 | 53s | -1.4 pp | 5 épocas |
| 2.4 lr = 1e-5 | 73.45% | 74.71% | 98,554 | 61s | -14.9 pp | 5 épocas |
| 2.5 dropout = 0,0 | 89.00% | 88.69% | 98,554 | 54s | -0.9 pp | 5 épocas |
| 2.5 dropout = 0,3 | 88.30% | 88.17% | 98,554 | 52s | -1.4 pp | 5 épocas |
| 2.5 dropout = 0,7 | 82.90% | 82.28% | 98,554 | 78s | -7.3 pp | 5 épocas |
| 2.6 sem BatchNorm, lr=1e-3 | 86.15% | 86.46% | 98,442 | 71s | -3.1 pp | 5 épocas |
| 2.6 sem BatchNorm, lr=1e-2 | 84.07% | 84.13% | 98,442 | 70s | -5.4 pp | 5 épocas |

### 2.1 Sem normalização

Retirar o `Normalize` não impediu o aprendizado (87,68% de teste vs. 88,00% de validação — coerentes entre si depois da correção do bug de paridade descrito na metodologia), mas custou **~1,9 pp** frente à configuração equivalente com normalização (2.4, `lr=1e-3`, 88,17%). Entradas em `[0, 1]` em vez de média ~0 fazem o otimizador dar passos mais desiguais entre pesos — a rede ainda converge, só que um pouco mais devagar dentro do mesmo orçamento de 5 épocas.

### 2.2 Sem augmentation

Contra-intuitivamente, sem augmentation o teste ficou **melhor** (90,11% vs. 88,17% da configuração equivalente com augmentation) dentro de apenas 5 épocas — e a distância entre treino e validação ficou pequena (treino 90,57%, validação 90,92%). Isso não é evidência de que augmentation atrapalha: em 5 épocas, augmentation reduz a "quantidade efetiva" de cada imagem vista sem distorção, então o modelo aprende mais devagar mas de forma mais robusta. A curva completa da linha de base (12 épocas, com augmentation) mostra o padrão oposto acontecendo com mais tempo: perda de validação sempre abaixo da de treino, sem sinal de subida — ou seja, o custo de curto prazo do augmentation vem com um ganho de generalização que só aparece com mais épocas.

### 2.3 Capacidade do modelo

Canais (8,16,32) — 44.226 parâmetros, **4,5× menos** que a base — perderam 2,8 pp de teste (85,37%). Canais (32,64,128) — 241.770 parâmetros, 2,5× mais — ganharam apenas 0,5 pp sobre o equivalente de 5 épocas (88,69% vs. 88,17%), quase o dobro do tempo por época (84,7s vs. 52,8s). **Mais parâmetros não ajudou proporcionalmente**: a rede maior tem capacidade de sobra para um problema de 28×28 em escala de cinza, então o gargalo deixou de ser capacidade e passou a ser dado/tempo de treino — dobrar o tempo de época para meio ponto percentual é uma troca ruim aqui.

### 2.4 Taxa de aprendizado

`lr=1e-3` (88,17%) foi o único das três a convergir de forma estável. `lr=1e-1` desabou para 54,29% — grande demais: a acurácia de validação oscilou entre épocas (chegou a 52,5% no meio, caiu para 33,2% na última) porque cada passo do otimizador ultrapassa o mínimo em vez de se aproximar dele. `lr=1e-5` ficou em 74,71% — pequeno demais para os pesos se moverem de forma significativa em apenas 5 épocas (a perda de treino mal se distanciou do ponto de partida); com mais épocas ele chegaria a valores razoáveis, só que gastando um múltiplo do tempo.

### 2.5 Dropout

A diferença treino−validação cresce monotonicamente com o dropout: `−0,91 pp` em 0,0 → `−2,96 pp` em 0,3 → `−5,35 pp` em 0,7 (valores negativos porque o treino usa augmentation e a validação não — ver nota abaixo). O que confirma o papel regularizador não é o sinal do gap, mas o quanto ele cresce: mais dropout torna o treino sistematicamente mais difícil que a validação, que é exatamente o efeito esperado de desligar neurônios ao acaso. Em compensação, `dropout=0,7` foi longe demais para apenas 5 épocas — acurácia de teste caiu para 82,28%, pior entre os três; regularização forte exige mais tempo de treino para compensar a informação perdida a cada passo.

**Por que o gap é negativo mesmo sem overfitting:** o treino usa `RandomHorizontalFlip`/`RandomCrop`/`RandomRotation`, a validação não — então o treino está sistematicamente resolvendo um problema mais difícil que a validação. Um gap negativo pequeno é saudável; o que se observa aqui é como esse gap *cresce* com o dropout, não seu sinal absoluto.

### 2.6 Ablação do BatchNorm

Sem BatchNorm, `lr=1e-3` (o mesmo da base) rendeu 86,46% — 1,7 pp abaixo do equivalente com BatchNorm (88,17%). Subir para `lr=1e-2` sem BatchNorm **piorou ainda mais** (84,13%), o oposto do que aconteceria com BatchNorm ligado. Isso é consistente com o papel do BatchNorm descrito no material da aula: ele estabiliza a distribuição das ativações entre camadas, o que é justamente o que permite usar taxas de aprendizado maiores sem divergir. Sem essa estabilização, `lr` maior amplifica o problema em vez de acelerar a convergência.

## Nível 3 — Implementação

### 3.1 Baseline honesto — MLP vs. CNN

`MLPSimples` (achatar 784 → densa 128 → ReLU → densa 10) treinado nas mesmas 12 épocas, mesma semente, mesmos dados:

| Modelo | Parâmetros | Teste top-1 | Tempo/época |
| --- | --- | --- | --- |
| MLP (3.1) | 101.770 | 84,96% | ~30,5s |
| CNN (linha de base) | 98.554 | 89,58% | ~52,1s |

Este é o experimento que justifica a convolução: a CNN usa **3.216 parâmetros a menos** que o MLP e ainda assim acerta **4,6 pp a mais** no teste. O MLP gasta a maior parte do seu orçamento (100.352 parâmetros — quase o modelo inteiro da CNN) só na primeira camada, ligando cada um dos 784 pixels a cada um dos 128 neurônios ocultos, sem nenhuma noção de vizinhança espacial: um padrão de borda aprendido num canto da imagem não transfere para outro canto. A CNN aprende o mesmo tipo de padrão uma vez e reaplica em toda a imagem — daí acertar mais gastando menos.

![Curvas de perda e acurácia do MLP baseline, 12 épocas](assets/mlp-curvas.png)

Curvas do MLP: convergência mais rápida por época (menos computação por passo), mas platô mais baixo — evidência de menor capacidade representacional, não de falta de treino.

### 3.2 Matriz de confusão interpretada

![Matriz de confusão normalizada por linha da linha de base no conjunto de teste](assets/fashion-mnist-confusao.png)

Matriz de confusão da linha de base (10.000 imagens de teste), normalizada por linha.

| Par confundido | Confusões | Hipótese | Intervenção |
| --- | --- | --- | --- |
| Camisa ↔ Camiseta/Top | 232 | Ambas são peças de tronco fotografadas de frente; em 28×28 cinza, gola e comprimento de manga — o que as diferencia — são exatamente o detalhe perdido na baixa resolução | Resolução maior de entrada (56×56), ou crops focados na região de gola/manga |
| Camisa ↔ Casaco | 103 | Casaco vestido aberto tem silhueta parecida com camisa; textura de botão/lapela some em escala de cinza de baixa resolução | Aumentar profundidade dos blocos finais (campo receptivo maior capta padrões de botão/lapela), ou usar imagens coloridas |
| Bota ↔ Tênis | 104† | Bota cano-curto e tênis cano-alto convergem numa mesma silhueta a 28px de altura | Crops de perfil enfatizando altura do cano/sola; alternativa: fundir as duas classes se a aplicação não precisar da distinção fina |

† 54 Botas→Tênis + 50 Tênis→Bota, lido diretamente da matriz.

O padrão geral: os quatro piores F1 da linha de base (Camisa 0,66, Camiseta/Top 0,84, Casaco 0,84, Pulôver 0,86) formam um bloco quase fechado de confusão mútua — peças de tronco em tecido plano, sem textura de cor para ajudar. Calça, Bolsa e Sandália (F1 ≥ 0,96) quase nunca erram porque sua silhueta é distintiva mesmo em baixa resolução. A acurácia global de 89,58% esconde completamente essa estrutura — é por isso que a etapa de teste do pipeline original insiste em ir além da acurácia.

![Os doze erros de maior confiança da linha de base no conjunto de teste](assets/fashion-mnist-erros.png)

Os doze erros de maior confiança da linha de base — vários são peças de tronco ambíguas mesmo para um observador humano a 28×28 pixels.

### 3.3 Métrica de topo-2

Implementada em `evaluate.py` (`acuracia_topk`) e calculada automaticamente em toda avaliação deste projeto — não é um script à parte.

| Modelo | Top-1 | Top-2 | Ganho |
| --- | --- | --- | --- |
| Linha de base | 89,58% | 97,39% | +7,81 pp |
| 2.4 `lr=1e-1` (instável) | 54,29% | 76,97% | +22,68 pp |

O salto do modelo instável (+22,7 pp) é revelador: mesmo quando o treino não converge bem o suficiente para acertar de primeira, a classe certa quase sempre está entre as duas mais prováveis — o modelo "sabe aproximadamente", só não decide com segurança. **Aplicação real:** busca visual em e-commerce (fotografar uma peça de roupa e receber duas categorias sugeridas para o usuário confirmar com um toque) é um caso natural para top-2: dado que Camisa e Camiseta/Top são exatamente o par mais confundido do exercício 3.2, mostrar as duas opções para confirmação humana resolve com um clique uma ambiguidade que custaria 10+ pp de acurácia para o modelo resolver sozinho.

### 3.4 Visualização de filtros

![Os 16 filtros 3x3 aprendidos na primeira convolução](assets/filtros-conv1.png)

Os 16 filtros 3×3 da primeira convolução.

![Mapas de ativação do primeiro bloco convolucional para uma imagem de teste da classe Camisa](assets/ativacoes-bloco1.png)

Mapas de ativação do bloco 1 para uma imagem de teste da classe Camisa.

Nenhum desses filtros foi desenhado à mão — os 16 padrões acima emergiram só de retropropagação. Mesmo assim, o resultado é reconhecível: a maioria são **detectores de gradiente orientado** — pares claro/escuro em diagonais, horizontais ou verticais (f1, f4, f9, f12), o equivalente aprendido de um filtro de borda de Sobel. Alguns (f6, f10) parecem detectores de canto, sensíveis a mudanças de intensidade em duas direções ao mesmo tempo. Nos mapas de ativação, os canais c11 e c13 acendem fortemente exatamente no contorno da peça contra o fundo — resposta clássica de detector de borda aplicado à silhueta —, enquanto c2, c6 e c8 respondem às linhas verticais internas do tecido (dobras/costuras). É o padrão esperado para a primeira camada de qualquer CNN treinada em imagens: antes de reconhecer "o que é", a rede aprende "onde muda".

### 3.5 Robustez com fotos reais

**Não executado nesta rodada.** Este exercício depende de fotos reais de roupas (ou desenhos) fornecidas por quem está resolvendo os exercícios — nenhuma foi fornecida. O código está pronto e testado (`src/predict.py`, reaproveita a mesma verificação de paridade de pré-processamento discutida na metodologia):

```
python predict.py caminho/da/foto.jpg --tag baseline --inverter
```

Expectativa documentada no `README` original: como o Fashion-MNIST tem objeto claro sobre fundo escuro, uma foto comum (roupa escura sobre fundo claro) precisa da flag `--inverter`; sem ela, a rede vê algo estatisticamente muito diferente do que aprendeu e tende a errar com confiança — o mesmo fenômeno de distribuição fora do domínio de treino que aparece na discussão do Nível 4 abaixo, só que por causa de iluminação/fundo em vez de dataset inteiro.

## Nível 4 — Caminho A: CIFAR-10

Mesma arquitetura (três blocos Conv→BatchNorm→ReLU→MaxPool, mesmo `canais_conv=(16,32,64)`), só a configuração de dados mudou.

| Parâmetro | Fashion-MNIST | CIFAR-10 |
| --- | --- | --- |
| Canais / resolução | 1 × 28 × 28 | 3 × 32 × 32 |
| Normalização (média / desvio) | 0,286 / 0,353 | (0,491,0,482,0,447) / (0,247,0,244,0,262) |
| Flatten após os 3 blocos | 64×3×3 = 576 | 64×4×4 = 1.024 |
| Parâmetros treináveis | 98.554 | 156.186 |
| Épocas treinadas | 12 | 15 |
| Tempo mediano/época | ~52s | ~73s |

Nenhuma linha de código do modelo mudou — só `config_para_dataset("CIFAR10")`. O tamanho do flatten (1.024, contra 576) foi descoberto pela mesma técnica empírica do `model.py` original (passar um tensor de zeros pelo extrator), exatamente o motivo pelo qual esse design existe: trocar de dataset não exigiu recalcular nada à mão.

![Mosaico de imagens de treino do CIFAR-10, coloridas, com augmentation aplicado](assets/cifar10-mosaico.png)

Mosaico de treino do CIFAR-10 após augmentation — note o quanto mais informação (cor, textura, fundo) cada imagem carrega frente ao Fashion-MNIST.

![Curvas de perda e acurácia do modelo treinado no CIFAR-10, 15 épocas](assets/cifar10-curvas.png)

Curvas de treino, 15 épocas — perda de validação ainda caindo no fim, sem sinal de overfitting.

![Matriz de confusão normalizada por linha do modelo CIFAR-10 no conjunto de teste](assets/cifar10-confusao.png)

Matriz de confusão do CIFAR-10.

| Métrica | Fashion-MNIST | CIFAR-10 | Queda |
| --- | --- | --- | --- |
| Acurácia de teste (top-1) | 89,58% | 76,18% | −13,40 pp |
| Acurácia de teste (top-2) | 97,39% | 89,79% | −7,60 pp |
| F1 da pior classe | 0,66 (Camisa) | 0,59 (Gato) | −0,07 |

### Por que o CIFAR-10 é mais difícil

Quatro fatores concretos, visíveis nos próprios resultados acima:

**1. Silhueta deixa de bastar.** No Fashion-MNIST, o contorno sozinho quase resolve o problema (por isso Calça e Bolsa chegam a F1 ≥ 0,97 mesmo em escala de cinza). No CIFAR-10 o par mais confundido é **Cachorro → Gato** (19% dos cachorros) e **Gato → Cachorro** (13% dos gatos) — dois animais quadrúpedes de silhueta muito parecida, que só se distinguem por textura de pelagem, formato do focinho e orelha, detalhes finos que sobrevivem mal a 32×32 pixels.

**2. Fundo e pose variam; no Fashion-MNIST, não.** Toda imagem do Fashion-MNIST é a peça de roupa centralizada sobre fundo uniforme escuro. No CIFAR-10, cada foto tem fundo, iluminação e ângulo próprios — a rede precisa aprender a ignorar tudo isso, não só a reconhecer a classe.

**3. Mais variação intra-classe.** "Pássaro" no CIFAR-10 cobre dezenas de espécies, poses e distâncias de câmera; "Camiseta/Top" no Fashion-MNIST é sempre a mesma peça fotografada do mesmo jeito. F1 de Pássaro (0,65) e Veado (0,73) refletem exatamente essa dispersão.

**4. Mesma arquitetura, problema maior.** A rede que resolve Fashion-MNIST com folga (98.554 parâmetros, três blocos) não foi redesenhada para o CIFAR-10 — só recebeu mais um canal de entrada. Redes que competem de verdade nesse dataset (ResNets, camadas mais profundas, augmentation mais agressivo) chegam a 93–96%; a diferença entre 76% aqui e esse patamar é literalmente a diferença entre "adaptar a configuração" e "redesenhar a arquitetura", que é o ponto pedagógico deste caminho.

## Problemas e limitações

A primeira rodada do experimento 2.1 apresentou um erro de paridade de pré-processamento: `evaluate.py` normalizava o conjunto de teste mesmo quando o treino havia sido executado sem normalização. O resultado incorreto foi 27,6% de acurácia no teste, enquanto a validação estava em 88%. A avaliação foi corrigida para receber a mesma flag `--sem-normalizacao` usada no treino, e o experimento foi reavaliado. O resultado correto é 87,68%.

As doze ablações do Nível 2 usaram uma única semente e cinco épocas por configuração. Esse desenho é suficiente para observar a direção dos efeitos dos hiperparâmetros, mas não sustenta afirmações de significância estatística fina entre configurações próximas. O próprio material cita como exemplo a diferença de 0,52 pp entre `dropout=0,0` e `dropout=0,3`.

A máquina hibernou brevemente durante a rodada. Três das catorze execuções tiveram uma época com duração de dezenas de minutos por esse motivo. As tabelas usam a mediana do tempo por época para reduzir o efeito desses outliers; os tempos brutos permanecem em `outputs/<tag>/historico.json`.

O exercício 3.5, de robustez com fotos reais, não foi executado porque nenhuma imagem de entrada foi fornecida. O código de predição está pronto para essa validação.

## Conclusão

Os resultados justificam o uso de convoluções tanto na comparação com o MLP quanto na execução sobre Fashion-MNIST. A CNN de referência usa menos parâmetros que o MLP e alcança maior acurácia. As ablações mostram, dentro do orçamento de treino utilizado, os efeitos de normalização, augmentation, capacidade, taxa de aprendizado, dropout e BatchNorm.

No CIFAR-10, a mesma arquitetura caiu para 76,18% de acurácia top-1, uma redução de 13,40 pp em relação ao Fashion-MNIST. O resultado mostra a limitação de reutilizar uma arquitetura simples quando o problema passa a exigir informação de cor, textura, fundo e maior variação intra-classe.

O código completo está organizado em `cnn_junior/`. A estrutura e os comandos de reprodução estão documentados em `cnn_junior/README.md`.
