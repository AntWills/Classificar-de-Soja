# Classificador de Soja

## Identificação

- Nome do estudante: Antonio Wills da Silva Santos
- Disciplina: Aprendizado Profundo
- Modelo implementado: CNN
- Dataset utilizado: SoyBeans-Seed
- Link do dataset: https://data.mendeley.com/datasets/v6vzvfszj6/6

## Dataset
O conjunto de dados SoyBeans-Seed foi criado por Lin et al. (2023), e inclui cinco tipos de sementes de soja individuais, totalizando 5.513 amostras: intactas, manchadas, imaturas, quebradas e com danos na casca, com mais de 1.000 imagens por categoria. As imagens originais, contendo múltiplas sementes em contato físico, foram capturadas por uma câmera industrial (3072×2048 pixels) e posteriormente segmentadas em imagens de sementes individuais (227×227 pixels) por meio de um algoritmo de processamento de imagem com precisão de segmentação superior a 98%. O dataset é adequado tanto para tarefas de classificação quanto para estudos de avaliação da qualidade de sementes de soja. 

Os dados foram divididos em 70% para treinamento, 20% para validação e 10% para teste. Todas as imagens foram redimensionadas para 224×224 pixels. Para o conjunto de treinamento, foram aplicadas as seguintes técnicas de aumento de dados : remoção aleatória
de regiões da imagem (Random Erasing) e desfoque gaussiano aplicado
de forma probabilística, conforme mostrado no Código abaixo.

```python
transformacao_treino = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.25),
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(3, 5))
    ], p=0.5),
])
```

## Ambiente
- Linguagem: Python;
- Versão do Python: 3.12;
- Bibliotecas: pytorch, torchvision, sklearn, datasets, matplotlib. kagglehub. splitfolders. numpy. pathlib, random, shutil;
- Uso de GPU: sim.

## Modelo
Foi construída uma CNN composta por uma camada de entrada, quatro blocos convolucionais intercalados com max pooling, uma camada de pooling global adaptativa, uma etapa de achatamento
(flatten) e duas camadas densas finais, com dropout para
regularização. Os detalhes de cada etapa estão descritos a seguir.

```
RedeCNN(
  (convolucoes): Sequential(
    (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): ReLU()
    (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): ReLU()
    (8): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (9): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (10): ReLU()
    (11): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (12): AdaptiveAvgPool2d(output_size=(1, 1))
  )
  (classificador): Sequential(
    (0): Flatten(start_dim=1, end_dim=-1)
    (1): Linear(in_features=64, out_features=256, bias=True)
    (2): ReLU()
    (3): Dropout(p=0.3, inplace=False)
    (4): Linear(in_features=256, out_features=196, bias=True)
  )
)
```

## Principais resultados

Metricas do conjunto de testes:
- Acurácia: 90.29%
- Precisão (macro): 0.9004
- Recall (macro): 0.8989
- F1-Score (macro): 0.8989

## Como executar

Execute sequencialmente cada celula, não tem erro.