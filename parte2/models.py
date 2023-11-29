import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.
        
        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        Deberiais obtener el producto escalar (o producto punto) que es "equivalente" a la distancia del coseno
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(self.w,x)




    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.
        Dependiendo del valor del coseno devolvera 1 o -1
        
        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        score=nn.as_scalar(self.run(x))
        return 1 if score>=0 else -1




    def train(self, dataset):
        """
        Train the perceptron until convergence.
        Hasta que TODOS los ejemplos del train esten bien clasificados. Es decir, hasta que la clase predicha en se corresponda con la real en TODOS los ejemplos del train
        """
        "*** YOUR CODE HERE ***"
        #cambiar los np.asscalar a np.ndarray.item (en nn.y linea 392, y en autograder.py linea 338)

        notConverge=True
        while notConverge:
            notConverge=False
            for x,y in dataset.iterate_once(1):
                output=self.get_prediction(x)
                label=nn.as_scalar(y)
                if output!=label:
                    notConverge=True
                    self.w.update(x,label)
        




class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    NO ES CLASIFICACION, ES REGRESION. ES DECIR; APRENDER UNA FUNCION.
    SI ME DAN X TENGO QUE APRENDER A OBTENER LA MISMA Y QUE EN LA FUNCION ORIGINAL DE LA QUE QUIERO APRENDER
    """
    def __init__(self):

        "*** YOUR CODE HERE ***"            # CON ESTOS PARAMETROS LOSS=0.018129
        salida_tamaño=1         
        entrada_tamaño=1        
        oculta_tamaño=30       # tamaños de capa oculta (10-400)
        self.batch_size = 100    # tamaño del lote (1-200)
        self.lr = -0.001         # tasa de aprendizaje (0,001-1)
        self.numero_ocultas=3   # numero de capas ocultas (1-3)

        
        self.layers=[]

        # capa de entrada
        self.layers.append(nn.Parameter(entrada_tamaño,oculta_tamaño))
        self.layers.append(nn.Parameter(1, oculta_tamaño))

        # capas ocultas
        for i in range(self.numero_ocultas):
            self.layers.append(nn.Parameter(oculta_tamaño, oculta_tamaño))
            self.layers.append(nn.Parameter(1, oculta_tamaño))
        
        # capa de salida
        self.layers.append(nn.Parameter(oculta_tamaño,salida_tamaño))
        self.layers.append(nn.Parameter(1, salida_tamaño))
    

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1). En este caso cada ejemplo solo esta compuesto por un rasgo
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values.
            Como es un modelo de regresion, cada valor y tambien tendra un unico valor
        """
        "*** YOUR CODE HERE ***"
        
        # capa de entrada
        entrada = nn.Linear(x, self.layers[0])          # X*W0 (con 1 capa oculta)
        entrada = nn.AddBias(entrada, self.layers[1])   # (X*W0)+B0
        
        # Capas ocultas
        for i in range(2, len(self.layers)-2, 2):
            oculta = nn.Linear(nn.ReLU(entrada), self.layers[i]) # relu(X*W0+B0)*W1  
            oculta = nn.AddBias(oculta, self.layers[i + 1])      # relu(X*W0+B0)*W1+B1
            entrada = oculta

        # Capa de salida
        salida = nn.Linear(nn.ReLU(entrada), self.layers[-2])      # relu(relu(X*W0+B0)*W1+B1)*W2
        salida = nn.AddBias(salida, self.layers[-1])               # relu(relu(X*W0+B0)*W1+B1)*W2+B2

        return salida



    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
                ----> ES FACIL COPIA Y PEGA ESTO Y ANNADE LA VARIABLE QUE HACE FALTA PARA CALCULAR EL ERROR 
                return nn.SquareLoss(self.run(x),ANNADE LA VARIABLE QUE ES NECESARIA AQUI), para medir el error, necesitas comparar el resultado de tu prediccion con .... que?
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(x),y)

    #hay que cambiar la linea 392 de nn.py a return float(...)
    def train(self, dataset):
        """
        Trains the model.
        
        """
        batch_size = self.batch_size
        total_loss = 100000

        while total_loss > 0.02:
            #ITERAR SOBRE EL TRAIN EN LOTES MARCADOS POR EL BATCH SIZE COMO HABEIS HECHO EN LOS OTROS EJERCICIOS
            #ACTUALIZAR LOS PESOS EN BASE AL ERROR loss = self.get_loss(x, y) QUE RECORDAD QUE GENERA
            #UNA FUNCION DE LA LA CUAL SE  PUEDE CALCULAR LA DERIVADA (GRADIENTE)

            "*** YOUR CODE HERE ***"
            for x, y in dataset.iterate_once(batch_size):
                prediction=self.run(x)
                if prediction!=y:
                    total_loss=self.get_loss(x,y)
                    grad=nn.gradients(total_loss,self.layers)
                    total_loss=nn.as_scalar(total_loss)
                    for i in range(0, len(self.layers), 2):
                        self.layers[i].update(grad[i], self.lr)
                        self.layers[i + 1].update(grad[i + 1], self.lr)


            
class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        # TEN ENCUENTA QUE TIENES 10 CLASES, ASI QUE LA ULTIMA CAPA TENDRA UNA SALIDA DE 10 VALORES,
        # UN VALOR POR CADA CLASE

        output_size = 10 # TAMANO EQUIVALENTE AL NUMERO DE CLASES DADO QUE QUIERES OBTENER 10 CLASES
        pixel_dim_size = 28
        pixel_vector_length = pixel_dim_size* pixel_dim_size
 
        "*** YOUR CODE HERE ***"            # CON ESTOS PARAMETROS  10 EPOCH Y ACCURACY=97.42
        #salida_tamaño=1         # definida arriba, output size
        entrada_tamaño=784        
        oculta_tamaño=200        
        self.batch_size = 300    # tamaño del lote (1-tamaño data=60000)
        self.lr = -0.1         # tasa de aprendizaje (0,001-1)
        self.numero_ocultas=3   # numero de capas ocultas (1-3)

        
        self.layers=[]

        # capa de entrada
        self.layers.append(nn.Parameter(entrada_tamaño,oculta_tamaño))
        self.layers.append(nn.Parameter(1, oculta_tamaño))

        # capas ocultas
        for i in range(self.numero_ocultas):
            self.layers.append(nn.Parameter(oculta_tamaño, oculta_tamaño))
            self.layers.append(nn.Parameter(1, oculta_tamaño))
        
        # capa de salida
        self.layers.append(nn.Parameter(oculta_tamaño,output_size))
        self.layers.append(nn.Parameter(1, output_size))
        

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
            output_size = 10 # TAMANO EQUIVALENTE AL NUMERO DE CLASES DADO QUE QUIERES OBTENER 10 "COSENOS"
        """
        "*** YOUR CODE HERE ***"                
        # capa de entrada
        entrada = nn.Linear(x, self.layers[0])          
        entrada = nn.AddBias(entrada, self.layers[1])  
        
        # Capas ocultas
        for i in range(2, len(self.layers)-2, 2):
            oculta = nn.Linear(nn.ReLU(entrada), self.layers[i])  
            oculta = nn.AddBias(oculta, self.layers[i + 1])      
            entrada = oculta

        # Capa de salida
        salida = nn.Linear(nn.ReLU(entrada), self.layers[-2])      
        salida = nn.AddBias(salida, self.layers[-1])               

        return salida




    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).
        POR EJEMPLO: [0,0,0,0,0,1,0,0,0,0,0] seria la y correspondiente al 5
                     [0,1,0,0,0,0,0,0,0,0,0] seria la y correspondiente al 1

        EN ESTE CASO ESTAMOS HABLANDO DE MULTICLASS, ASI QUE TIENES QUE CALCULAR 
        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"#NO ES NECESARIO QUE LO IMPLEMENTEIS, SE OS DA HECHO
        return nn.SoftmaxLoss(self.run(x), y) # COMO VEIS LLAMA AL RUN PARA OBTENER POR CADA BATCH
                                              # LOS 10 VALORES DEL "COSENO". TENIENDO EL Y REAL POR CADA EJEMPLO
                                              # APLICA SOFTMAX PARA CALCULAR LA PROBABILIDA MAX
                                              # Y ESA SERA SU PREDICCION,
                                              # LA CLASE QUE MUESTRE EL MAYOR PROBABILIDAD, LA PREDICCION MAS PROBABLE, Y LUEGO LA COMPARARA CON Y 

    def train(self, dataset):
        """
        Trains the model.
        EN ESTE CASO EN VEZ DE PARAR CUANDO EL ERROR SEA MENOR QUE UN VALOR O NO HAYA ERROR (CONVERGENCIA),
        SE PUEDE HACER ALGO SIMILAR QUE ES EN NUMERO DE ACIERTOS. EL VALIDATION ACCURACY
        NO LO TENEIS QUE IMPLEMENTAR, PERO SABED QUE EMPLEA EL RESULTADO DEL SOFTMAX PARA CALCULAR
        EL NUM DE EJEMPLOS DEL TRAIN QUE SE HAN CLASIFICADO CORRECTAMENTE 
        """

        while dataset.get_validation_accuracy() < 0.975:
            for x, y in dataset.iterate_once(self.batch_size):
                prediction=self.run(x)
                if prediction!=y:
                    total_loss=self.get_loss(x,y)
                    grad=nn.gradients(total_loss,self.layers)
                    total_loss=nn.as_scalar(total_loss)
                    for i in range(0, len(self.layers), 2):
                        self.layers[i].update(grad[i], self.lr)
                        self.layers[i + 1].update(grad[i + 1], self.lr)








