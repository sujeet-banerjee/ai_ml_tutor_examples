from ipywidgets import interact

def greet(name):
    print ( f'Hello, {name}!')

interact(greet, name='Greet')
