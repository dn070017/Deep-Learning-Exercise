#%%
import numpy as np
import tensorflow as tf
import asyncio

from bokeh.models import Arrow, Button, ColumnDataSource, CustomJS, GlyphRenderer, VeeHead, Slider, Label, DataTable, TableColumn, RadioButtonGroup, Range1d, NumberFormatter, LinearColorMapper, ColorBar
from bokeh.palettes import Magma256
from bokeh.layouts import column, row
from bokeh.plotting import curdoc, figure, output_notebook, show
from copy import deepcopy
import time

#%%
def loss(w_hat: tf.Variable,  x_custom: tf.Tensor, y_custom: tf.Tensor, test: bool=False,) -> np.float32:
    l = tf.reduce_sum(tf.pow(y_custom - tf.matmul(x_custom, w_hat), 2), axis=0).numpy()
    if test:
        return l
    if num_dims.active == 0:
        return l.reshape((1, num_domain)) 
    elif num_dims.active == 1:
        return l.reshape((num_domain, num_domain))

def gradient(w_hat: tf.Variable, x_custom: tf.Tensor, y_custom: tf.Tensor) -> np.float32:
    with tf.GradientTape() as g:
        loss = tf.reduce_sum(tf.pow(y_custom - tf.matmul(x_custom, w_hat), 2), axis=0)
    return g.gradient(loss, w_hat).numpy()

def init():
    tf.random.set_seed(0)
    global x, w, w_hat, y, y_domain, w_domain, loss_surface, loss_source, num_domain
    
    num_domain = 500

    w = list()
    w.append([w1.value])
    w.append([w2.value])

    w_hat = list()
    w_hat.append([init_w1.value])
    w_hat.append([init_w2.value])
    w_hat = tf.Variable(w_hat[0:num_dims.active + 1], dtype=tf.dtypes.float32)

    x = tf.random.normal((num_samples.value, num_dims.active + 1), mean=0, stddev=1)
    w = tf.Variable(w[0:num_dims.active + 1], dtype=tf.dtypes.float32)
    y = tf.matmul(x, w)

    if num_dims.active == 0:
        w_domain = tf.reshape(tf.linspace(-25., 25., num_domain), (1, num_domain))
    elif num_dims.active == 1:
        w1_domain, w2_domain = np.meshgrid(np.linspace(-20., 20., num=num_domain),
                                           np.linspace(-20., 20., num=num_domain))
        w_domain = tf.convert_to_tensor(np.array([w1_domain.ravel(), w2_domain.ravel()], dtype=np.float32))
    y_domain = tf.matmul(x, w_domain)

def change_dim(value, old, new):
    if new == 0:
        w1.width = 500
        w1.title = "Ground Truth Weight"
        init_w1.width = 500
        init_w1.title = "Initial Weight"
        w2.visible = False
        init_w2.visible = False
        layout.children[0] = p1d
        metrics[1].visible = False
        metrics[0].visible = True
    elif new == 1:
        w1.width = 245
        w1.title = "Ground Truth Weight 1"
        init_w1.width = 245
        init_w1.title = "Initial Weight 1"
        w2.visible = True
        init_w2.visible = True
        layout.children[0] = p2d
        metrics[0].visible = False
        metrics[1].visible = True

    change_params(None, None, None)

def change_params(value, old, new):
    init()
    batch_size.end = num_samples.value
    loss_surface = loss(w_domain, x, y)  
    l = loss(w_hat, x, y, test=True)
    g = gradient(w_hat, x, y)
    
    if num_dims.active == 0:
        loss_source.data = dict(w=w_domain.numpy(), L=loss_surface)
        past_w_source[0].data = dict(w=[], L=[], g=[])
        now_w_source[0].data = dict(e=[0], w=[init_w1.value], L=[l[0]])
        past_w_source[1].data = dict(w1=[], w2=[], L=[], g=[])
        now_w_source[1].data = dict(e=[0], w1=[init_w1.value], w2=[init_w2.value], L=[l[0]])
    elif num_dims.active == 1:
        loss_surface = loss(w_domain, x, y) 
        p2d.select_one({'name': 'loss'}).data_source.data['image'] = [loss_surface]
        past_w_source[0].data = dict(w=[], L=[], g=[])
        now_w_source[0].data = dict(e=[0], w=[init_w1.value], L=[l[0]])
        past_w_source[1].data = dict(w1=[], w2=[], L=[], g=[])
        now_w_source[1].data = dict(e=[0], w1=[init_w1.value], w2=[init_w2.value], L=[l[0]])

    p1d.center = p1d.center[0:2]
    p2d.center = p2d.center[0:2]
    update()

def restart():
    change_params(None, None, None)

def update():
    if now_w_source[num_dims.active].data['e'][0] + 1 > max_epoch.value:
        return

    w_hat = list()
    if num_dims.active == 0:
        w_hat.append([now_w_source[0].data['w'][0]])
    elif num_dims.active == 1:
        w_hat.append([now_w_source[1].data['w1'][0]])
        w_hat.append([now_w_source[1].data['w2'][0]])  
    w_hat = tf.Variable(w_hat, dtype=tf.dtypes.float32)

    batches = [batch_size.value] * (num_samples.value // batch_size.value)
    if num_samples.value % batch_size.value != 0:
        batches.extend([num_samples.value % batch_size.value])

    for batch_x, batch_y in zip(tf.split(x, batches, axis=0), tf.split(y, batches, axis=0)):
        
        g = gradient(w_hat, batch_x, batch_y)
        
        w_hat_old = w_hat.numpy()
        w_hat = tf.Variable(w_hat - eta.value * g)
        l = loss(w_hat, x, y, test=True)

        if num_dims.active == 0:
            arrow = Arrow(end=VeeHead(size=5), line_color='black',
                          x_start=w_hat_old[0, 0],
                          x_end=w_hat.numpy()[0, 0],
                          y_start=0, y_end=0)
            p1d.add_layout(arrow)

            new_data = dict()
            new_data['w'] = past_w_source[0].data['w']
            new_data['L'] = past_w_source[0].data['L']
            new_data['w'].append(w_hat.numpy()[0, 0])
            new_data['L'].append(l[0])
            
            if len(new_data['w']) > 10:
                new_data['w'] = new_data['w'][-11:]
                new_data['L']  = new_data['L'][-11:]

            past_w_source[0].data = new_data
        elif num_dims.active == 1:
            arrow = Arrow(end=VeeHead(size=5), line_color='black',
                          x_start=w_hat_old[0, 0], 
                          y_start=w_hat_old[1, 0], 
                          x_end=w_hat.numpy()[0, 0], 
                          y_end=w_hat.numpy()[1, 0])
            p2d.add_layout(arrow)

            new_data = dict()
            new_data['w1'] = past_w_source[1].data['w1']
            new_data['w2'] = past_w_source[1].data['w2']
            new_data['L']  = past_w_source[1].data['L']
            new_data['w1'].append(w_hat.numpy()[0, 0])
            new_data['w2'].append(w_hat.numpy()[1, 0])
            new_data['L'].append(l[0])
            if len(new_data['w1']) > 10:
                new_data['w1'] = new_data['w1'][-11:]
                new_data['w2'] = new_data['w2'][-11:]
                new_data['L']  = new_data['L'][-11:]
            past_w_source[1].data = new_data
        
    if num_dims.active == 0:
        now_w_source[0].data = dict(e=[now_w_source[0].data['e'][0] + 1], w=[w_hat.numpy()[0, 0]], L=[l[0]])
        for arrow in p1d.center[2:-(len(batches))]:
            arrow.line_color = 'lightgrey'
            arrow.end.fill_color  = 'lightgrey'
            arrow.end.line_color  = 'lightgrey'
        if len(p1d.center) > 12:
            p1d.center = p1d.center[0:2] + p1d.center[-10:]

    else:
        now_w_source[1].data = dict(e=[now_w_source[1].data['e'][0] + 1], w1=[w_hat.numpy()[0, 0]], w2=[w_hat.numpy()[1, 0]], L=[l[0]], g=[g[0]])
        for arrow in p2d.center[2:-(len(batches))]:
            arrow.line_color = 'lightgrey'
            arrow.end.fill_color  = 'lightgrey'
            arrow.end.line_color  = 'lightgrey'
        if len(p2d.center) > 12:
            p2d.center = p2d.center[0:2] + p2d.center[-10:]
    #print(w_hat)

# %%
button = Button(label='Restart', align='center', max_width=500)
button.on_click(restart)
num_dims = RadioButtonGroup(labels=['1-D', '2-D'], active=0)
num_dims.on_change('active', change_dim)

w1 = Slider(start=-20, end=20, value=-10, step=0.1, title="Ground Truth Weight", align='center', width=500)
w2 = Slider(start=-20, end=20, value=-10, step=0.1, title="Ground Truth Weight 2", align='center', width=245)
init_w1 = Slider(start=-20, end=20, value=5, step=0.1, title="Initial Weight", align='center', width=500)
init_w2 = Slider(start=-20, end=20, value=5, step=0.1, title="Initial Weight 2", align='center', width=245)
w2.visible = False
init_w2.visible = False
w1.on_change('value', change_params)
w2.on_change('value', change_params)
init_w1.on_change('value', change_params)
init_w2.on_change('value', change_params)

num_samples = Slider(start=4, end=20, value=4, step=1, title="Number of Samples", align='center', width=500)
num_samples.on_change('value', change_params)
batch_size = Slider(start=1, end=num_samples.value, value=2, step=1, title="Batch Size", align='center', width=500)
batch_size.on_change('value', change_params)
eta = Slider(start=0.01, end=0.5, value=0.01, step=0.01, title="Learning Rate", align='center', width=500)
eta.on_change('value', change_params)
max_epoch = Slider(start=1, end=50, value=25, step=1, title="Number of Epochs", align='center', width=500)
#max_epoch.on_change('value', update_init)

init()
loss_surface = loss(w_domain, x, y)
loss_source = ColumnDataSource(data=dict(w=w_domain.numpy(), L=loss_surface))
g = gradient(w_hat, x, y)[0]
l = loss(w_hat, x, y, test=True)[0]

past_w_source = dict()
past_w_source[0] = ColumnDataSource(data=dict(w=[], L=[]))
past_w_source[1] = ColumnDataSource(data=dict(w1=[], w2=[], L=[]))

now_w_source = dict()
now_w_source[0] = ColumnDataSource(data=dict(e=[0], w=[init_w1.value], L=[l]))
now_w_source[1] = ColumnDataSource(data=dict(e=[0], w1=[init_w1.value], w2=[init_w2.value], L=[l]))

p1d = figure(tools=['save'],
             title='Gradient Descent 1D',
             plot_width=720, plot_height=480, 
             x_axis_label="w",
             y_axis_label="L",
             x_range=(-21, 21),
             y_range=(-50, np.max(loss_surface)))
p1d.line(x='w', y='L', color='lightgrey', source=loss_source)
p1d.circle(x='w', y='L', color='lightgrey', source=past_w_source[0])
p1d.circle(x='w', y='L', color='black', source=now_w_source[0])

p2d = figure(tools=['save'],
             title='Gradient Descent 2D',
             plot_width=720, plot_height=480, 
             x_axis_label='w1', y_axis_label='w2',
             x_range=(-20, 20), y_range=(-20, 20))

color_mapper = LinearColorMapper(Magma256[::-1], low=np.min(loss_surface), high=np.max(loss_surface))
color_bar = ColorBar(color_mapper=color_mapper, border_line_color=None, location=(0,0), label_standoff=8, title='Loss', scale_alpha=0.5)
p2d.add_layout(color_bar, 'right')
p2d.image(image=[loss_surface], name='loss',
          x=-20, y=-20, dw=40, dh=40, alpha=0.5, color_mapper=color_mapper)
p2d.circle(x='w1', y='w2', color='lightgrey', source=past_w_source[1])
p2d.circle(x='w1', y='w2', color='black', source=now_w_source[1])

columns = dict()
columns[0] = [TableColumn(field="e", title="Epoch", formatter=NumberFormatter(text_align='right')),
              TableColumn(field="w", title="Weight", formatter=NumberFormatter(text_align='right', format='0,0.0000')),
              TableColumn(field="L", title="Loss", formatter=NumberFormatter(text_align='right', format='0,0.0000'))]
columns[1] = [TableColumn(field="e", title="Epoch", formatter=NumberFormatter(text_align='right')),
              TableColumn(field="w1", title="Weight 1", formatter=NumberFormatter(text_align='right', format='0,0.0000')),
              TableColumn(field="w2", title="Weight 2", formatter=NumberFormatter(text_align='right', format='0,0.0000')),
              TableColumn(field="L", title="Loss", formatter=NumberFormatter(text_align='right', format='0,0.0000'))]

metrics = dict()
metrics[0] = DataTable(source=now_w_source[0], columns=columns[0], width=500, height=50, index_position=None, align='center')
metrics[1] = DataTable(source=now_w_source[1], columns=columns[1], width=500, height=50, index_position=None, align='center')
metrics[1].visible = False

layout = row(p1d, column(num_dims, row(w1, w2), row(init_w1, init_w2), num_samples, batch_size, eta, max_epoch, metrics[0], metrics[1], button))
curdoc().add_root(layout)
curdoc().add_periodic_callback(update, 1000)
