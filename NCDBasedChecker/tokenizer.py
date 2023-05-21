from lib2to3.pgen2.token import STRING
import antlr4
from antlr4 import *
from LexersAndParsers.Java9 import Java9Lexer,Java9Listener,Java9Parser
import re
import numpy as np
import io
import tokenize
from io import BytesIO
import token
from pyminifier import obfuscate
from pyminifier import token_utils
class JavaTokenizer:

    def __init__(self,filePath,code=None,filterTokens=set()):
        if(filePath):
            with open(filePath,"r") as f:
                self.code = f.read()
        else:
            self.code=code

        self.lexer = Java9Lexer.Java9Lexer(antlr4.InputStream(self.code))
        self.stream = antlr4.CommonTokenStream(self.lexer)
        self.stream.getText()
        #self.parser = Java9Parser.Java9Parser(self.stream)
        #self.tree = self.parser.compilationUnit()
        #print(tree.toStringTree(recog=parser))
        self.TokenNumber_to_TokenType=self.generateDict()
        if(len(filterTokens)==0):
            self.filterToks=None
            self.filteredDict={ k:v  for k,v in self.TokenNumber_to_TokenType.items() if(k>=11 and k<119)}
        else:
            self.filterToks=filterTokens
            self.filteredDict={{ k:v  for k,v in self.TokenNumber_to_TokenType.items() if(v in filterTokens)}}

    def getTokenVector(self):
        num_calls=self.getNumMethodCalls()
        freq_dict=dict()
        for toks in self.stream.tokens:
            if(toks.type in self.filteredDict):
                #print(self.filteredDict)
                if(toks.type in freq_dict):
                    freq_dict[toks.type]+=1
                else:
                    freq_dict[toks.type]=0

        vec_size=len(self.filteredDict.values())+1
        vec= np.zeros((1,vec_size))
        vec[0,vec_size-1]=num_calls
        for i,k in enumerate(sorted(list(freq_dict.keys()))):
            vec[0,i]=freq_dict[k]
        return vec

    
    def getNumMethodCalls(self):
        regex= r'[a-zA-Z]+\([^\)]*\)(\.[^\)]*\))?'
        m =re.finditer(regex,self.code)
        #newcode=self.code
        matches=[ma for ma in m]
        # for i,match in enumerate(reversed(matches[1:])):
        #     newcode= newcode[0:match.span(0)[0]] + "METHODCALL("+match[0]+")" +newcode[match.span(0)[1]:]   
        return len(matches)

    def getTokenizedCode(self):
        """
        returns two strings: first being the tokenized code and second being the original code with tabs stripped
        """
        #self.stream.getText()
        tokens=[]
        for toks in self.stream.tokens:
            tokens.append((toks.text,toks.type,self.TokenNumber_to_TokenType[toks.type],toks.start,toks.stop,toks.text))
        newCode=[]
        oldCode=[]
        #print(tokens)
        for i,toks in enumerate(tokens):
            if(not (toks[1]==117 or toks[1]==118)):
                newCode.append(toks[2]+" ")
                oldCode.append(toks[5]+" ")
            #print(tokens[i])
  
        return "".join(newCode),"".join(oldCode)


    def generateDict(self):
        tokens=open('./LexersAndParsers/Java9/Java9Lexer.tokens','r').read().split('\n') #update file location for new projects
        number_to_type={}
        for token in tokens[0:119]:
            pair=token.rsplit('=',1)
            number_to_type[int(pair[1])]=pair[0]
        number_to_type[-1]='EOF'
        number_to_type[0]='PAD'
        return number_to_type
    
    # def remove_comments(self,string):
    #     pattern = r"(\".*?\"|\'.*?\')|(/\*.*?\*/|//[^\r\n]*$)"
    #     # first group captures quoted strings (double or single)
    #     # second group captures comments (//single-line or /* multi-line */)
    #     regex = re.compile(pattern, re.MULTILINE|re.DOTALL)
    #     def _replacer(match):
    #         # if the 2nd group (capturing comments) is not None,
    #         # it means we have captured a non-quoted (real) comment string.
    #         if match.group(2) is not None:
    #             return "" # so we will return empty to remove the comment
    #         else: # otherwise, we will return the 1st group
    #             return match.group(1) # captured quoted-string
    #     return regex.sub(_replacer, string)
    def remove_comments(self,string):
        string = re.sub(re.compile("/\*.*?\*/",re.DOTALL ) ,"" ,string) # remove all occurrences streamed comments (/*COMMENT */) from string
        string = re.sub(re.compile("//.*?\n" ) ,"" ,string) # remove all occurrence single-line comments (//COMMENT\n ) from string
        return string


class PythonTokenizer:
    def __init__(self,filePath,code=None,name_generator=None,filterTokens=set()) :
        if(filePath):
            with open(filePath,"r") as f:
                self.code = f.read()
        else:
            self.code=code
        self.name_generator = name_generator
        
    def getTokenizedCode(self,obfuscate_var=False):
        out=self.remove_comments_and_docstrings(self.code)
        if obfuscate_var:
            out= self.obfuscate_variables(out)
       
        g = tokenize.tokenize(BytesIO(out.encode()).readline)
            
        res=['']
        old=[] 
        for tup in g:
            toknum,tok_val,_,_,_=tup
            if(toknum!=token.TYPE_COMMENT and toknum!=token.COMMENT and toknum!=token.NL and toknum!=token.NEWLINE):
                if  (tok_val.startswith("'''") or tok_val.startswith('"""')) and res[-1]!='EQUAL ':
                    continue

                if(toknum==token.NAME or toknum==token.STRING):
                    res.append(tok_val+" ")
                elif toknum==token.OP:
                    res.append(token.tok_name[tup.exact_type]+" ")
                else:
                    res.append(token.tok_name[toknum]+" ")
                old.append(tok_val +" ")
        return "".join(res[1:]), "".join(old)
    
    def obfuscate_variables(self,code):
        options= dotdict()
        options.replacement_length=5
        options.obf_variables =True
        options.obfuscate=False
        toks=token_utils.listified_tokenizer(code)
        obfuscate.obfuscate(module="Q2C116",tokens=toks , options=options, name_generator = self.name_generator)
        return token_utils.untokenize(toks)
    

    def remove_comments_and_docstrings(self,source):
        io_obj = io.StringIO(source)
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            ltext = tok[4]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += (" " * (start_col - last_col))
            if token_type == tokenize.COMMENT:
                pass
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
                    if prev_toktype != tokenize.NEWLINE:
                        if start_col > 0:
                            out += token_string
            else:
                out += token_string
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
        out = '\n'.join(l for l in out.splitlines() if l.strip())
        return out

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    _delattr__ = dict.__delitem__
