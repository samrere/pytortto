from test import *

def create():
    c('def forward(x):')
    with indent():
        c('sum=0')
        c('for i in x:')
        with indent():
            c('sum+=i\nreturn sum')
    finished()
if __name__ == '__main__':
    create()



