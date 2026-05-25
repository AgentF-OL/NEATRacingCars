# TODO - Remove this file since this game does not use decision tree based cars

import random

### Decision Trees

""" An object oriented implementation of a decision tree
No __init__ method in the most general class
"""

# Classe mais abstract... everything is a node
class Node(object):
    """ Abstract Tree Root implementation
    """

    def decide(self, info):
        raise NotImplementedError


# Behavior or action nodes
# which also "decide# but the decision method returns the action name.
class Action(Node):
    """ nome da acção
    """
    def __init__(self,name):
        self.name=name

    def decide(self, _):
        return self.name


# Decision Nodes
class Decision(Node):
    """ Abstract interior node implementation
    """

    def __init__(self, attr, daughters):
        self._attribute = attr
        self._daughter_nodes = daughters

    def decide(self, info):
        return self.getBranch(info).decide(info)
    
    def value(self,info):
        return info[self._attribute]

    def getBranch(self,info):
        return self._daughter_nodes[self.value(info)]


## Boolean Decision Nodes
class Boolean(Decision):
    """ Abstract boolean decision node implementation
    """

    def __init__(self, attr,yesNode,noNode):
        super(Boolean, self).__init__(attr,{True: yesNode, False: noNode})

## Boolean Less Than
class Less(Decision):
    """ Abstract range decision node implementation,extending Boolean node
    """

    def __init__(self, attr,yesNode,noNode,max):
        super(Less, self).__init__(attr,{True: yesNode, False: noNode})
        self.maxValue = max

    def value(self,info):
        return info[self._attribute] < self.maxValue

## Boolean Greater Than
class Greater(Decision):
    """ Abstract range decision node implementation,extending Boolean node
    """

    def __init__(self, attr,yesNode,noNode,max):
        super(Greater, self).__init__(attr,{True: yesNode, False: noNode})
        self.maxValue = max

    def value(self,info):
        return info[self._attribute] > self.minValue

## Boolean within a interval
class MinMax(Decision):
    """ Abstract range decision node implementation,extending Boolean node
    """

    def __init__(self, attr,yesNode,noNode,min, max):
        super(MinMax, self).__init__(attr,{True: yesNode, False: noNode})
        self.minValue = min
        self.maxValue = max

    def value(self,info):
        return self.maxValue >= info[self._attribute] >= self.minValue

class RandomDecision(Decision):
    """ A Random Decision--- uniform 
    """
    def __init__(self,daughters):
        #super(Decision, self).__init__(attr,daughters)
        self._daughter_nodes=daughters
        
    def decide(self,info):
        d=random.choice(self._daughter_nodes)
        return d.decide(info)

## Choose randomly, stuck with a branch for a while and change branch randomly
## Change (do not conserve the last branch)
class RandomDecisionPeriod(Decision):
    """ A Random Decision--- uniform -during a certain number of ticks or frames or calls 
    """
    def __init__(self,daughters,lim):
        #super(Decision, self).__init__(attr,daughters)
        self._daughter_nodes=daughters
        self.lim=lim
        self.ticks=lim
        self.current_decision=random.choice(self._daughter_nodes)
        
    def decide(self,info):
        if  self.ticks <= 0:
            new_dec=random.choice(self._daughter_nodes)
            while new_dec==self.current_decision:
                new_dec=random.choice(self._daughter_nodes)
            self.current_decision=new_dec
            self.ticks=self.lim-1
        self.ticks-=1
        return self.current_decision.decide(info)


##-----------------  AgentDT

class DTAgent(object):
    
    def __init__(self,name,dt):
        self.name=name
        self.sensors=None
        self.dt=dt
    
    def update_sensors(self):
        raise NotImplementedError
    
    def decision(self):
        action=(self.dt).decide(self.sensors)
        eval('self.'+action+'()')


        
