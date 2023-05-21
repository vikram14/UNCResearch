class Submission:

    def __init__(self,code,student_info,subID):
        self.code=code
        self.studentName=student_info[0]
        self.studentId=student_info[1]
        self.submissionId=subID

    def setfilePath(self,path):
        self.filePath=path
    
    def getfilePath(self):
        return self.filePath

    def getstudentName(self):
        return self.studentName
    
    def getstudentID(self):
        return self.studentId
