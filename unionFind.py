class UnionFind:
    def __init__(self,student_Ids:list) -> None:
        self.numberOfStudents : int  = len(student_Ids)
        self.studentIdToArrayIndex : dict = {k:i for i, k in enumerate(student_Ids)}
        self.arrayIndexToStudentId : dict = {i:k for i, k in enumerate(student_Ids)}
        # parent[i] is the parent of the student i. parent[0] is a dummy cell for clarity (no student 0)
        self.parent : list = [i for i in range(self.numberOfStudents+1)]
        # initially each student is in a group himself. Can be used to find size of cluster.
        self.sizes : list = [1 for i in range(self.numberOfStudents+1)]
        # number of clusteres
        self.numberOfGroups = self.numberOfStudents
    
    def find(self,student_id:str)->str:
        root : int= self.studentIdToArrayIndex[student_id]
        id : int = root
        # find root
        while(root!=self.parent[root]):
            root = self.parent[root]
        # path compression
        while(self.parent[id]!=id):
            temp : int = self.parent[id]
            self.parent[id] = root
            id = temp

        return self.arrayIndexToStudentId[root]
    
    def union(self,student_id1:str, studentid_2:str) -> None:
        root_1 : int = self.studentIdToArrayIndex[self.find(student_id1)]
        root_2 : int = self.studentIdToArrayIndex[self.find(studentid_2)]
        if (root_1 == root_2):
            return
        if(self.sizes[root_1]<self.sizes[root_2]):
            # Join trees for root 1 into root 2
            self.sizes[root_2]+=self.sizes[root_1]
            self.parent[root_1]=root_2
        else:
            # Join trees for root 2 into root 1
            self.sizes[root_1]+=self.sizes[root_2]
            self.parent[root_2] = root_1

        self.numberOfGroups -=1

        return

    def size(self,student_id:str):
        # returns size of group this student id belongs to
        return self.sizes[self.studentIdToArrayIndex[self.find(student_id)]]

    def getNumberOfGroups(self):
        return self.numberOfGroups

        

        
