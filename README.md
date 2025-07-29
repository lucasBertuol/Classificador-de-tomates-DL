# Classificador-de-tomates-DL
Este projeto foi desenvolvido como estudo de caso didático para avaliar e comparar o desempenho de abordagens de **transfer learning** com redes neurais de Deep Learning treinadas do zero na tarefa de reconhecimento de **maturação de tomates** em imagens. Essa é uma atividade que faz parte do Bootcamp  BairesDev - Machine Learning Training oferecido pela [DIO](https://www.dio.me/) e foi realizada a partir de um tutorial para transfer learning disponibilizado pelo meu professor acessível nesse [link](https://colab.research.google.com/github/kylemath/ml4a-guides/blob/master/notebooks/transfer-learning.ipynb#scrollTo=VWWN-FPLYoZs) .

##  Objetivos 🎯
 - Demonstrar, com exemplos práticos, como o transfer learning pode superar ou igualar modelos criados do zero, especialmente em cenários com poucos dados. A tarefa central é classificar imagens em duas categorias: **tomates maduros** e **tomates verdes**.
 - Comparar diferentes abordagens de **Transfer learning** e **Feature extraction**
 
 ## Estrutura do projeto 🗂️
 -   **Tomato_Classifier_1.ipynb:**  Transfer learning com apenas a camada de saída substituída; (~8.000 parâmetros treináveis)
    
-   **Tomato_Classifier_2.ipynb:**  Transfer learning com head customizado;  (~3 milhões de parâmetros treináveis)
    
-   **Ambos notebooks incluem também a arquitetura e treinamento de um modelo do zero para comparação**
    
-   **dataset/train/**  e  **dataset/test/**: Imagens organizadas por classe: ripe (maduro) e unripe (verde)
    
-   **Imagens dos resultados:**  Gráficos de loss e acurácia ao longo de 20 epochs de treinamento.
## Recursos Utilizados ⚒️

-   **Linguagem:**  Python 3.x 
    
-   **Framework:**  TensorFlow (Keras API)
    
-   **Execução:**  Google Colab (Possui bibliotecas necessárias já instaladas e interface amigável. Executei em runtime local para maior velocidade)
    
-   **Modelo base para Transfer Learning:**  rede VGG16 pré-treinada em ImageNet (diretamente via keras.applications)
- **Dataset com imagens de tomates**
## Dataset 📊
Eu utilizei um dataset do Kaggle já pronto contendo imagens de tomates em vários estágios de maturação, disponível nesse [link](https://www.kaggle.com/datasets/sumn2u/riped-and-unriped-tomato-dataset). É fundamental garantir que os dados estejam rotulados e organizados corretamente, e que haja um equilíbrio razoável entre as classes para evitar vieses no aprendizado. Por isso, os dados estão organizados dessa forma:

    Dataset/
	    Test/ # 110 no total
		    Ripe/ # 55 imagens
		    Unripe/ # 55 imagens
		    
	    Train/ # 329 no total
		    Ripe/ # 155 imagens
		    Unripe/ # 174 imagens
Algumas centenas de imagens já são suficientes para treinar a nossa rede, por se tratar de uma tarefa de classificação simples, mas o ideal seria possuir mais imagens. Por isso, vamos aplicar **data augmentation** para melhorar a qualidade do nosso dataset. 
## Metodologia 📖
### Data augmentation 
Eu apliquei esse método para artificialmente aumentar o **tamanho** e a **variedade** do conjunto de dados de treino (ds_treino). O objetivo é melhorar a robustez e previnir o overfitting (quando o modelo memoriza os dados de treino em vez de aprender padrões gerais), aumentando a capacidade de **generalização** dos modelos. Foram aleatoriamente alterados a orientação, rotação, zoom e contraste das imagens. 

### Transfer learning
Transfer learning é uma técnica de Machine Learning em que aproveitamos o **aprendizado** de um modelo de IA para treinar outro modelo. Ao construir um modelo do zero, é necessário possuir um dataset com uma grande quantidade e variabilidade de dados para obter bons resultados. Por outro lado, ao utilizar um modelo pré-treinado podemos usar o conhecimento adquirido em determinada tarefa e aplicar a um novo problema relacionado. Dessa forma, economizam-se recursos computacionais e se torna possível alcançar uma boa **precisão** mesmo com um dataset limitado, o que é muito útil para o nosso caso.  
Dentro do Transfer Learning, existem duas estratégias principais:  **Feature extraction**  e  **Fine-tuning**. Essas estratégias são consideradas subdivisões ou abordagens distintas dentro do Transfer Learning.

-   **Feature extraction**: utiliza-se o modelo pré-treinado como um extrator de features, aproveitando as representações aprendidas nas primeiras camadas e, geralmente, apenas treinando novas camadas "head" (de classificação) para a tarefa desejada. Mais eficaz para datasets pequenos. 
    
-   **Fine-tuning**: além de adicionar ou treinar o head, parte ou todas as camadas do modelo pré-treinado passam por um ajuste fino (ou seja, são retreinadas), permitindo ao modelo adaptar-se melhor à nova tarefa ou domínio. É necessário possuir dados suficientes para não causar overfitting. 

Devido ao dataset pequeno, eu optei por realizar Feature extraction. Entretanto, eu não tinha certeza se as features de alto nível do ImageNet seriam relevantes para um caso específico como o de maturação de tomates, apesar de se tratarem de objetos do mundo real. Por essa razão, foram exploradas duas formas distintas de realizar a Feature Extraction:
### Método 1: substituir a camada de saída da rede pré-treinada
Esse método foi usado no arquivo **Tomato_Classifier_1**. Nele eu treino novamente a camada de decisão final da VGG16, adaptando-a para as minhas classes. Apenas essa última camada da rede permance treinável, o resto da rede fica cogelada.  A vantagem dessa estratégia é aproveitar as features genéricas do ImageNet, treinar mais rapidamente e reduzir o risco de overfitting com o nosso dataset pequeno. Em contrapartida, a adaptação do modelo fica limitada se as features pré-existentes não forem suficientemente representativas para a minha tarefa.

**Processo:**
1. Eu importei a rede VGG16 com pesos pré-treinados em ImageNet(`weights='imagenet'`), e com o head original `include_top=True`
2. A seguir, eu criei uma nova camada densa para o meu número de classes (2), usando softmax como ativação. 
3. Então, troquei a camada de classificação final da VGG16, uma camada softmax de 1000 neurônios correspondente à ImageNet, por essa nova camada de 2 neurônios. Com isso,  crei uma nova rede chamada `model_new` 
4. O próximo ajuste foi congelar todas as camadas da rede `model_new`, exceto a última. Para compilar o modelo, usei a função de perda `categorical_crossentropy` (adequada para medir a performance em tarefas de classificação), otimizador `adam` e métricas para `accuracy`

imagem do model.summary()
### Método 2: treinar um novo classificador para o modelo
Esse método foi usado no arquivo **Tomato_Classifier_2**. As features de alto nível do modelo pré-treinado podem não ser relevantes para classificar tomates. É por isso que dessa vez eu utilizo o meu próprio head customizado, mais complexo e com mais parâmetros para se adaptar à minha tarefa específica. Simultaneamente, as features mais genéricas de camadas anteriores da VGG16 servem como base para o novo classificador. Essa rede pode aprender combinações mais ricas e não lineares das features extraídas, o que pode levar a um melhor desempenho, mas também aumenta o risco de overfitting se os dados forem insuficientes. Todos os blocos convolucionais do modelo são congelados, apenas o novo head é treinável. 

**Processo**:
1. Importei a VGG16 novamente com pesos pré-treinados mas dessa vez sem o head original `include_top=False`
2. Importei o resto da rede VGG16 com pesos congelados `vgg(inputs, training=False)`
3. Criei um head customizado para a minha tarefa de classificação (camada Flatten + Dense + Dropout) e uma nova camada de saída (com 2 neurônios). Nomeei a nova rede como `model_new`
4. Compilei o modelo da mesma forma, usando `loss='categorical_crossentropy'`, `optmizer='adam'` e `metrics='accuracy'`
Detalhes do modelo:

imagem do model.summary()

Em todos os modelos eu usei um batch_size de 128 e treinei por 20 epochs. 
### Modelo de controle: rede neural treinada do zero
Com a intenção de comparar a eficiência e os resultados de se utilizar Transfer learning em relação com um modelo de IA sem nenhum conhecimento prévio, eu desenvolvi um modelo chamado `model_scratch`. Detalhes do modelo:

imagem do model.summary()
## Resultados 📈
A seguir, estão os gráficos de **perda** e **precisão** obtidos com cada modelo. Em azul, o modelo treinado do zero e em laranja os respectivos modelos com Transfer Learning. 

**Tomato_classifier_1**

imagem Tomato_Classifier_1_output 

84.55% de precisão com 0.44 de perda no modelo de Transfer Learning contra 60.91% de precisão com 1.20 de perda no modelo treinado do zero. 

**Tomato_classifier_2**

imagem Tomato_Classifier_2_output 

81.82% de precisão com 0.76 de perda no modelo de Transfer Learning contra 66.36% de precisão com 0.75 de perda no modelo treinado do zero. 

Observando os gráficos é possível concluir que o modelo treinado do zero (`model_scratch`) não foi capaz de realizar um treinamento robusto mesmo com seus 1.2 milhões de neurônios e manteve uma precisão relativamente constante ao longo das 20 epochs. Ao aplicar o Transfer Learning, conseguimos observar uma melhora instantânea nos resultados do modelo, em decorrer do conhecimento prévio da rede pré-treinada sendo utilizada.
Por outra perspectiva, analisamos que apesar do modelo usado em Tomato_Classifier_1 ser **menos complexo**, com apenas 8.194 neurônios, ele foi capaz de atingir uma **melhor performance** entre todos os modelos. Em contrapartida, o modelo mais sofisticado empregado em Tomato_Classifier_2 apesar de possuir um número muito maior de neurônios treináveis não foi capaz de alcançar uma melhor acurácia. Ao se tornar mais complexa, essa rede se tornou mais suscetível ao overfitting. 

Classifier_2_overfitting
A precisão nos dados de treino aumentou enquanto nos dados de teste a precisão diminuiu nas últimas epochs.

Isso nos leva à conclusão de que nem sempre um modelo com mais parâmetros será o mais eficaz, pois em geral quanto maior a quantidade de parâmetros maior é a quantidade de dados necessária para alimentar o modelo. 
## Conclusão 🍅
Por fim, conclui-se que mesmo com um dataset pequeno é possível melhorar a acurácia de uma rede de Deep learning ao utilizar **Data augmentation** e **Transfer Learning**. Com apenas **0,68%** dos neurônios usados na rede feita do zero, obtemos uma maior precisão. Isso nos permite economizar tempo e recursos computacionais. Comparando as redes 1 e 2 constatamos que as features extraídas pela rede VGG16 são suficientemente **relevantes**, tornando o classificador mais simples do Tomato_Classifier_1 a abordagem mais **eficaz** para este problema específico. 
Para escolher a melhor estratégia de treinamento, é essencial considerar o **tamanho** do seu dataset e o **nível de similaridade** com a rede pré-treinada de sua escolha, pois é isso que irá definir o melhor caminho para conquistar bons resultados com Transfer learning.  
## Contato 📧
Estou aberto a contribuições e sugestões de melhoria!

Email: lucascondessabertuol@gmail.com

LinkedIn: https://www.linkedin.com/in/lucasbertuol/
