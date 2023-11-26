
import numpy as np
class NN():
   
    
    def __init__(self):
        np.random.seed(1)
        self.Weight1=np.random.random((3,14))
        self.Weight2=np.random.random((14,1))

    def sigmoid_derv(self, x):
      return(x*(1-x))
  
    def sigmoid(self,x):
      return (1/(1+np.exp(-x)))


    def train(self,X,t):
      Loss=[]
      
      for i in range(1000):
        alpha=5
        z=self.sigmoid(np.dot(X,self.Weight1))
        y=self.sigmoid(np.dot(z,self.Weight2))
        loss=1/4*np.sum((y-t)**2)
        grad_Weight2=2*(np.dot(y.T,(y-t)*self.sigmoid_derv(y)))
        grad_Weight1=2*np.dot(X.T,np.dot((y-t)*self.sigmoid_derv(y),self.Weight2.T)*self.sigmoid_derv(z))
        self.Weight2=self.Weight2-alpha*grad_Weight2
        self.Weight1=self.Weight1-alpha*grad_Weight1
        Loss.append(loss)

      for i in range(1000,20000):
        alpha=1
        z=self.sigmoid(np.dot(X,self.Weight1))
        y=self.sigmoid(np.dot(z,self.Weight2))
        loss=1/4*np.sum((y-t)**2)
        grad_Weight2=2*(np.dot(y.T,(y-t)*self.sigmoid_derv(y)))
        grad_Weight1=2*np.dot(X.T,np.dot((y-t)*self.sigmoid_derv(y),self.Weight2.T)*self.sigmoid_derv(z))
        self.Weight2=self.Weight2-alpha*grad_Weight2
        self.Weight1=self.Weight1-alpha*grad_Weight1
        Loss.append(loss)
     
    
    def think(self, inputs):        
        inputs = inputs.astype(float)
        hidden_layer=self.sigmoid(np.dot(inputs,self.Weight1))
        final_output=self.sigmoid(np.dot(hidden_layer,self.Weight2))
        return final_output
    

if __name__ == "__main__":
   nn = NN()
   print(nn.Weight1)
   print(nn.Weight2)

   inputs=np.vstack(([0,0,1],[1,0,1],[1,1,1]))
   outputs=np.array([1,0,1]).reshape(-1,1)
   

   nn.train(inputs,outputs)

   print("Synaptic weights after training: ")
   print(nn.Weight2)
   print(nn.Weight1)
   A = str(input("Input 1: "))
   B = str(input("Input 2: "))
   C = str(input("Input 3: "))
  
   print("New situation: input data = ", A, B, C)
   print("Output data: ")
   print(nn.think(np.array([A, B, C])))
    




    

  


