"""
Plotting the WaveNet model using Graphviz
"""

from graphviz import Digraph

def draw_residual_block_graphviz():
    dot = Digraph(comment='WaveNet Residual Block')
    
    # Set graph attributes
    dot.attr(rankdir='LR', size='10,5')

    # Define nodes
    dot.node('A', 'Input', shape='box')
    dot.node('B', 'Dilated 1D Convolution\nNo. of Kernels: 32\nDilation: (i-1)^2\nKernel size: 3', shape='box', style='filled', fillcolor='#ff9999')
    dot.node('C', 'Dilated 1D Convolution\nNo. of Kernels: 32\nDilation: (i-1)^2\nKernel size: 3', shape='box', style='filled', fillcolor='#ff9999')
    dot.node('D', 'tanh', shape='circle', style='filled', fillcolor='#ffff99')
    dot.node('E', 'σ', shape='circle', style='filled', fillcolor='#ffff99')
    dot.node('F', '×', shape='point')
    dot.node('G', '1 x 1 Convolution', shape='box', style='filled', fillcolor='#90ee90')
    dot.node('H', 'Residual', shape='box')
    dot.node('I', 'Skipped Connections', shape='box')

    # Define edges
    dot.edge('A', 'B', label='', arrowhead='normal')
    dot.edge('A', 'C', label='', arrowhead='normal')
    dot.edge('B', 'D', label='', arrowhead='normal')
    dot.edge('C', 'E', label='', arrowhead='normal')
    dot.edge('D', 'F', label='', arrowhead='normal')
    dot.edge('E', 'F', label='', arrowhead='normal')
    dot.edge('F', 'G', label='', arrowhead='normal')
    dot.edge('G', 'H', label='', arrowhead='normal')
    dot.edge('G', 'I', label='', arrowhead='normal')

    dot.edge('A', 'H', constraint='false')

    # Render graph
    dot.render('wavenet_residual_block', format='png', view=True)



def draw_wavenet_model_graphviz():
    dot = Digraph(comment='WaveNet Model')

    # Set graph attributes
    dot.attr(rankdir='LR', size='16,8')
    dot.attr('node', fontsize='16', width='2', height='1')

    # Define nodes for the first row
    dot.node('A', 'Raw Audio Waveform\n(32 Channels)', shape='box')
    dot.node('B', '1st Residual Block', shape='box', style='filled', fillcolor='#add8e6')
    dot.node('C', '2nd Residual Block', shape='box', style='filled', fillcolor='#add8e6')
    dot.node('D', '6th Residual Block', shape='box', style='filled', fillcolor='#add8e6')
    dot.node('E', '1 x 1 Convolution', shape='box', style='filled', fillcolor='#90ee90')
    
    # Define nodes for the second row
    dot.node('F', 'ReLU', shape='ellipse', style='filled', fillcolor='#ffff99')
    dot.node('G', 'Fully Connected (10)\nDropout 50%', shape='box', style='filled', fillcolor='#ff9999')
    dot.node('I', 'Softmax', shape='ellipse', style='filled', fillcolor='#ffff99')
    dot.node('J', 'EGG Signal Waveform', shape='box')

    # Arrange nodes in the first row from left to right
    dot.edge('A', 'B')
    dot.edge('B', 'C')
    dot.edge('C', 'D')
    dot.edge('D', 'E')
    
    # Arrange nodes in the second row from right to left
    dot.edge('J', 'I')
    dot.edge('I', 'G')
    dot.edge('G', 'F')
    dot.edge('F', 'E')

    # Add skipped connections from first row to second row
    dot.edge('B', 'E', style='dashed')
    dot.edge('C', 'E', style='dashed')
    dot.edge('D', 'E', style='dashed')

    # Render graph
    dot.render('wavenet_model_horizontal_two_rows_corrected_v2', format='png', view=True)

draw_wavenet_model_graphviz()
