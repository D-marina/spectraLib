#!/usr/bin/env python
# coding: utf-8

# In[28]:


class Mesh1d:
    def __init__(self,a,b,ncells):
        self.nodes = [(b-a)/ncells*i for i in range(ncells)]
        self.vertex = [[i,i+1] for i in range(0,ncells)]
        print(self.nodes,self.vertex)
    def getNodos(self):
        return self.nodes
    def getVertex(self):
        return self.vertex


# In[29]:


malla = Mesh1d(0,1,4)


# In[30]:


malla.getNodos()


# In[31]:


malla.getVertex()

