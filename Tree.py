from sklearn import tree
from random import randint

class DecisionTree():
    def test(self):
        Tests = [[1,0,1,0,1,1,0],
        [1,0,0,0,1,0,1],
        [0,0,1,1,0,1,1],
        [0,1,0,0,1,1,0],
        [0,1,1,0,1,0,1],
        [0,1,0,1,0,1,1],
        [0,1,0,0,1,0,0],
        [0,0,1,1,1,1,1],
        [0,0,1,0,0,0,1],
        [1,1,1,1,1,1,1],
        [0,1,0,1,1,0,1],
        [0,0,0,1,1,0,1],
        [1,0,1,0,1,1,1],
        [0,0,0,0,1,1,0],
        [0,0,1,0,0,1,0],
        [1,1,1,0,0,0,0],
        [0,0,0,0,1,1,0],
        [0,0,1,0,1,0,0],
        [1,1,0,0,1,1,0],
        [1,0,1,1,1,0,0],
        [0,1,0,0,0,0,0],
        [0,0,1,1,0,1,0],
        [0,1,1,0,1,0,0],
        [1,1,0,1,1,1,0],
        [1,1,1,1,1,0,0],
        [0,1,0,0,0,1,1],
        [0,1,0,1,0,0,0],
        [0,1,1,1,1,0,1],
        [1,1,1,1,1,1,0],
        [1,1,1,1,1,0,1],
        [1,0,1,0,0,0,1],
        [1,0,1,1,1,1,0],
        [1,0,0,0,0,0,1],
        [1,1,1,1,0,0,1],
        [0,0,0,0,0,0,1],
        [0,1,1,0,1,0,1],
        [1,1,0,1,1,1,1],
        [0,1,1,1,0,0,0],
        [1,0,1,0,1,1,0],
        [1,1,0,1,1,1,0],
        [0,1,1,1,1,1,0],
        [1,0,1,1,1,1,1],
        [1,0,0,0,1,1,0],
        [1,1,1,0,1,1,1],
        [0,0,0,0,1,0,1],
        [1,0,1,1,0,0,1],
        [1,1,1,0,1,1,1],
        [0,1,1,0,0,1,1],
        [0,1,0,0,0,0,1],
        [0,1,0,0,1,1,1],
        [1,1,1,1,1,0,0],
        [1,1,0,0,1,1,1],
        [0,1,1,1,0,1,1],
        [0,1,0,1,1,0,1],
        [1,1,1,0,0,0,0],
        [0,0,0,0,0,0,1],
        [0,1,0,0,0,1,0],
        [0,0,1,0,1,0,1],
        [0,1,1,1,1,0,1],
        [0,1,0,0,1,1,1],
        [1,1,1,0,0,0,0],
        [1,0,1,0,1,1,1],
        [1,1,0,1,1,0,0],
        [1,1,1,0,0,0,0],
        [1,1,0,0,1,0,0],
        [0,1,0,0,0,1,1],
        [1,0,0,0,0,1,0],
        [1,1,0,0,0,1,0],
        [0,0,1,1,0,0,1],
        [1,0,0,1,0,0,0],
        [0,0,0,0,1,1,0],
        [0,0,0,0,1,0,0],
        [0,0,0,0,1,0,1],]

        TestScore=[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1]

        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(Tests,TestScore)

        Condition1=randint(0,30)
        if (Condition1==0):
            Condition1=1
        else: Condition1=0
        Condition2=randint(0,30)
        if (Condition2==0):
            Condition2=1
        else: Condition2=0
        Condition3=randint(0,30)
        if (Condition3==0):
            Condition3=1
        else: Condition3=0
        Condition4=randint(0,30)
        if (Condition4==0):
            Condition4=1
        else: Condition4=0
        Condition5=randint(0,1)
        if (Condition5==0):
            Condition5=1
        else: Condition5=0
        Condition6=randint(0,30)
        if (Condition6==0):
            Condition6=1
        else: Condition6=0
        Condition7=randint(0,1)
        if (Condition7==0):
            Condition7=1
        else: Condition7=0

        print("Czy cywile: %d" % Condition1)
        print("Czy sniezyca: %d" % Condition2)
        print("Czy burza: %d" % Condition3)
        print("Czy wichura: %d" % Condition4)
        print("Czy wymaga rozbrojenia: %d" % Condition5)
        print("Czy saper posiada wystarczajace umiejetnosci: %d" % Condition6)
        print("Czy jest w stanie rozbroic pomimo braku umiejetnosci: %d" % Condition7)
        print("Czy ostatecznie rozbrajamy?: %d" % clf.predict([[Condition1,Condition2,Condition3,Condition4,Condition5,Condition6,Condition7]])[0])

        return clf.predict([[Condition1,Condition2,Condition3,Condition4,Condition5,Condition6,Condition7]])[0]

# output: 0-come later, 1-defuse the bomb

