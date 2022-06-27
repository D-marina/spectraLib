import numpy as np  
import numpy.polynomial.legendre as npl
import sympy as sp
import matplotlib.pyplot as plt
from scipy.special import legendre
from scipy.interpolate import lagrange
import math 

def Transformation(nodes, weight,a,b):
    nodes=(b-a)/2*nodes+(a+b)/2
    weight=(b-a)/2 *weight
    return (nodes,weight)

def chop(lista):
    for i in range(len(lista)):
        if(abs(lista[i])<(10**(-10))):
            lista[i]=0
    return lista

class Cuadrature:
    def __init__(self,nodes=None,weight=None):
        if(list(nodes)!=None):
                self.nodes=nodes
        if(list(weight)!=None):
                self.weight=weight
        
    def getNodes(self):
        return self.nodes
    def getWeight(self):
        return self.weight
    def getNumPoints(self):
        return len(self.nodes)
    
    def Integrate(self,f):
        return self.Integrate_List(f(np.array(self.nodes)))
    
    def Integrate_List(self,lista):
        if (len(lista)==len(self.weight)):
            return np.array(lista).dot(np.array(self.weight))
        else:
            raise Exception("No has metido correctamente la lista")

class Lobato(Cuadrature):
    def __init__(self, num_points,a,b):
        aux=[-1,*sorted(list(legendre(num_points).deriv().r)),1]
        Cuadrature.__init__(self, nodes=np.array(aux), weight=np.array(2/(num_points*(num_points+1)*(legendre(num_points)(aux))**2)))
        self.nodes,self.weight=Transformation(self.nodes, self.weight, a, b)
    def __str__(self):
        return "Nodos: %s \nPesos: %s " %(self.nodes,self.weight)

class Legendre(Cuadrature):
    def __init__(self,num_points,a,b):
        Cuadrature.__init__(self,nodes=npl.leggauss(num_points)[0], weight=npl.leggauss(num_points)[1]) 
        self.nodes,self.weight=Transformation(self.nodes, self.weight, a, b)
    def __str__(self):
        return "Nodos: %s \nPesos: %s " %(self.nodes,self.weight)

class Trapecio_Compuesto(Cuadrature): #están considerados los extremos(-1,1)
    def __init__(self,num_points,a,b):
        aux=2/num_points
        Cuadrature.__init__(self, nodes=np.array([*np.arange(-1,1,float(aux)).tolist(),1]), 
                            weight=np.array([2/(aux),*[aux]*(num_points-1),2/(aux)]))
        self.nodes,self.weight=Transformation(self.nodes, self.weight, a, b)
    def __str__(self):
        return "Nodos: %s \nPesos: %s " %(self.nodes,self.weight)

def Funciones_Base(N):
    '''
    Crea las funciones bases de grado N y devuelve un vector con las funciones bases y otro con sus derivadas
    '''
    x = sp.symbols('x')
    xvals = chop(Lobato(N,-1,1).getNodes())
    
    aux=np.zeros(N+1)
    phi=[]
    difphi=[]
    
    for i in range(0,N+1):
        aux[i]=1
        phi.append((lambda x:lagrange(xvals,aux))(x))
        difphi.append((lambda x: lagrange(xvals,aux).deriv())(x))  
        aux=np.zeros(N+1)
    return (phi,difphi)

def M_Difusion(m,p,Nq,num_interval):
    '''
    Función que construye la matriz local en una dimensión que integra la derivada de funciones bases en su respectivo intervalo
    '''
    intervals=np.linspace(-1,1,m+1)
    A=np.zeros((p+1, p+1))
    a=intervals[num_interval-1]
    b=intervals[num_interval]
    xvals2 = chop(Lobato(Nq-1,-1,1).getNodes())
    phi,difphi=Funciones_Base(p)
    J=(b-a)/2
    
    for i in range(0,p+1):
        for j in range(0,p+1):
            A[i][j]=Lobato(Nq-1,a,b).Integrate_List((difphi[i](xvals2))/J*(difphi[j](xvals2)/J))
    
    return A

def M_Masa(m,p,Nq,num_interval):
    '''
    Función que construye la matriz local en una dimensión que integra las funciones bases en su respectivo intervalo
    '''
    intervals=np.linspace(-1,1,m+1)
    A=np.zeros((p+1, p+1))
    a=intervals[num_interval-1]
    b=intervals[num_interval]
    xvals2 = chop(Lobato(Nq-1,-1,1).getNodes())
    phi,difphi=Funciones_Base(p)
    
    for i in range(0,p+1):
        for j in range(0,p+1):
            A[i][j]=Lobato(Nq-1,a,b).Integrate_List(phi[i](xvals2)*phi[j](xvals2))
    
    return A

def V_Indep(m,p,Nq,num_interval,f): 
    '''
    Función que construye el vector independiente en su respectivo intervalo de la función unidimensional y la funciones bases
    '''
    intervals=np.linspace(-1,1,m+1)
    A=np.zeros(p+1)
    a=intervals[num_interval-1]
    b=intervals[num_interval]
    phi,difphi=Funciones_Base(p)
 
    xvals2 = chop(Lobato(Nq-1,-1,1).getNodes())
    xvals3 = chop(Lobato(Nq-1,a,b).getNodes())
    
    for i in range(0,p+1):
            A[i]=Lobato(Nq-1,a,b).Integrate_List(f(xvals3)*phi[i](xvals2))  
        
    return A

def M_Masa_Global_1D(m,p,Nq):#m es nº de intervalos
    
    AG=np.zeros((p*m+1, p*m+1))
    for k in range(0,m):
        A=M_Masa(m,p,Nq,k+1)
        for i in range(0,p+1):
            for j in range(0,p+1):
                AG[k*p+i][k*p+j] = AG[k*p+i][k*p+j] + A[i][j]
    return AG

def M_Difusion_Global_1D(m,p,Nq):#m es nº de intervalos
    
    AG=np.zeros((p*m+1, p*m+1))
    for k in range(0,m):
        A=M_Difusion(m,p,Nq,k+1)
        for i in range(0,p+1):
            for j in range(0,p+1):
                AG[k*p+i][k*p+j] = AG[k*p+i][k*p+j] + A[i][j]
    return AG

def V_Indep_Global_1D(m,p,Nq,f):#m es nº de intervalos
    
    BG=np.zeros(p*m+1)
    for k in range(0,m):
        B=V_Indep(m,p,Nq,k+1,f)
        for i in range(0,p+1):
            BG[k*p+i]=BG[k*p+i]+B[i]
    return BG

def Solucion_1D(N_vals,p, Nq, f, cond1, cond2):
    AG=M_Difusion_Global_1D(N_vals,p,Nq) 
    BG=V_Indep_Global_1D(N_vals,p,Nq,f) 
    
    AG[0][0]=10**30
    BG[0]=cond1*10**30
    AG[len(AG)-1][len(AG)-1]=10**30
    BG[len(BG)-1]=cond2*10**30

    return np.linalg.solve(AG,BG)

def Indices(k,m):
    '''
    A partir de un número k (nodo k) obtenemos los índices (i,j) para saber a qué posición nos referimos
    Con la variable m indicamos el número de filas y columnas de la matriz total(consideramos matrices cuadradas)
    '''
    i=k%(m+1)
    j=k//(m+1)
    return (i,j)

def M_Global_2D(m,p,Nq):
    
    D=M_Difusion_Global_1D(m,p,Nq)
    M=M_Masa_Global_1D(m,p,Nq)
    A=np.zeros(((m*p+1)*(m*p+1),(m*p+1)*(m*p+1)))
    
    for k1 in range(0,(m*p+1)*(m*p+1)):
        i1,j1=Indices(k1,m*p)
        #print(" ")
        #print("k1: ",k1, "i1,j1: ",i1,j1)
        for k2 in range(0,(m*p+1)*(m*p+1)):
            i2,j2=Indices(k2,m*p)
            A[k1][k2]=D[i1][i2]*M[j1][j2]+M[i1][i2]*D[j1][j2]
                        
    return A

def V_Indep_Global_2D(m,p,Nq,f1,f2): 
    
    B=V_Indep_Global_1D(m,p,Nq,f1)
    B1=V_Indep_Global_1D(m,p,Nq,f2)
    B2=np.zeros((m*p+1)*(m*p+1))
    
    for k in range(0,(m*p+1)*(m*p+1)):
        i1,j1=Indices(k,m*p)
        #print("i1,j1",i1,j1)
        B2[k]=B[i1]*B1[j1]
                        
    return B2

def bloquear(A,B,m,p):
    for k in range(0,(m*p+1)**2):
        (i,j)=Indices(k,p*m)
        if (i==0 or i==p*m or j==0 or j==p*m):
            A[k][k]=10**30
            B[k]=0
    return (A,B)

def Solucion_2D(m,p,Nq,f1,f2):
    A=M_Global_2D(m,p,Nq)
    B=V_Indep_Global_2D(m,p,Nq,f1,f2)
    A,B=bloquear(A,B,m,p)
    U=np.linalg.solve(A,B)
    
    return U