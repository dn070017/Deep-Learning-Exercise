#!/usr/bin/env python
# coding: utf-8

# In[1]:


import colorcet
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import time

from bokeh.io import output_notebook, export_png, output_file, reset_output
from bokeh.layouts import row, column
from bokeh.palettes import linear_palette
from bokeh.plotting import figure, output_file, show, curdoc, ColumnDataSource
from bokeh.models import Range1d, HoverTool, CustomJS, Slider, ColorBar, LinearColorMapper, CategoricalColorMapper, Arrow, VeeHead, GlyphRenderer, Toggle
from copy import copy

reset_output()
output_notebook()

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# In[2]:


def l2_loss(pred, y):
    return (y - pred) ** 2
l2_loss = np.vectorize(l2_loss)


# In[3]:


def gradient(x, w, y):
    tx = tf.Variable(initial_value=x)
    tw = tf.Variable(initial_value=w)
    ty = tf.Variable(initial_value=y)
    with tf.GradientTape() as tape:
        loss = tf.sqrt(tf.reduce_mean((ty - tf.matmul(tx, tw)) ** 2))
    
    loss_grad = tape.gradient(loss, tw)
    return loss, loss_grad.numpy()


# In[4]:


def update_eta(attr, old, new):
    global eta, refresh
    eta = eta_slider.value
    refresh.active = True
    
def update_gamma(attr, old, new):
    global gamma, refresh
    gamma = gamma_slider.value
    refresh.active = True
    
def update_epsilon(attr, old, new):
    global epsilon, refresh
    epsilon = epsilon_slider.value
    refresh.active = True
    
def update_beta(attr, old, new):
    global beta, refresh
    beta = beta_slider.value
    refresh.active = True
    
def update_beta2(attr, old, new):
    global beta2, refresh
    beta2 = beta2_slider.value
    refresh.active = True

def update_x(attr, old, new):
    global x_init
    x_init = float(x_slider.value)
    refresh.active = True

def update_y(attr, old, new):
    global y_init
    y_init = float(y_slider.value)
    refresh.active = True


# In[5]:


def update():
    global m, v, p, epoch, pos, run, refresh, source
    epoch += 1
    if run.active and epoch <= num_epoch:
        run.label = 'Pause'
        # Batch Gradient Descent
        p['Batch Gradient Descent'].title.text = f'Batch Gradient Descent (η={eta:.1f}, epoch={epoch})'
        l, l_grad = gradient(x, pos['Batch Gradient Descent'], y)
        new_source_data = dict(x = [source['Batch Gradient Descent'].data['x'][-1] - eta * l_grad[0, 0]], y = [source['Batch Gradient Descent'].data['y'][-1] - eta * l_grad[1, 0]])
        source['Batch Gradient Descent'].stream(new_source_data, rollover=15)
        pos['Batch Gradient Descent'][0, 0] -= eta * l_grad[0, 0]
        pos['Batch Gradient Descent'][1, 0] -= eta * l_grad[1, 0]
        
        # Momemtum
        p['Momemtum'].title.text = f'Momemtum (η={eta:.2f}, 𝛾={gamma:.2f}, epoch={epoch})'
        l, l_grad = gradient(x, pos['Momemtum'], y)
        v['Momemtum'][0, 0] = gamma * v['Momemtum'][0, 0] + eta * l_grad[0, 0]
        v['Momemtum'][1, 0] = gamma * v['Momemtum'][1, 0] + eta * l_grad[1, 0]
        new_source_data = dict(x = [source['Momemtum'].data['x'][-1] - v['Momemtum'][0, 0]], y = [source['Momemtum'].data['y'][-1] - v['Momemtum'][1, 0]])
        source['Momemtum'].stream(new_source_data, rollover=15)
        pos['Momemtum'][0, 0] -= v['Momemtum'][0, 0]
        pos['Momemtum'][1, 0] -= v['Momemtum'][1, 0]
       
        # Nesterov Accelerate Gradient
        p['Nesterov Accelerate Gradient'].title.text = f'Nesterov Accelerate Gradient (η={eta:.1f}, 𝛾={gamma:.1f}, epoch={epoch})'
        
        t1 = v['Nesterov Accelerate Gradient'][:, 1]
        t2 = v['Nesterov Accelerate Gradient'][:, 0]
        v['Nesterov Accelerate Gradient'][:, 0] = t1
        v['Nesterov Accelerate Gradient'][:, 1] = pos['Nesterov Accelerate Gradient'].reshape(-1) + gamma * (t1 - t2)
        l, l_grad = gradient(x, v['Nesterov Accelerate Gradient'][:, 1].reshape(2, 1), y)
        new_source_data = dict(x = [v['Nesterov Accelerate Gradient'][0, 1] - eta * l_grad[0, 0]], 
                               y = [v['Nesterov Accelerate Gradient'][1, 1] - eta * l_grad[1, 0]])
        source['Nesterov Accelerate Gradient'].stream(new_source_data, rollover=15)
        pos['Nesterov Accelerate Gradient'][0, 0] = new_source_data['x'][0]
        pos['Nesterov Accelerate Gradient'][1, 0] = new_source_data['y'][0]
        
        # Adaptive Gradient Descent
        p['Adaptive Gradient Descent'].title.text = f'Adaptive Gradient Descent (η={eta:.1f}, epoch={epoch})'
        l, l_grad = gradient(x, pos['Adaptive Gradient Descent'], y)
        new_source_data = dict(x = [source['Adaptive Gradient Descent'].data['x'][-1] - eta / np.sqrt(epsilon + v['Adaptive Gradient Descent'][0, 0] ** 2 / epoch) * l_grad[0, 0]], 
                               y = [source['Adaptive Gradient Descent'].data['y'][-1] - eta / np.sqrt(epsilon + v['Adaptive Gradient Descent'][1, 0] ** 2 / epoch) * l_grad[1, 0]])
        source['Adaptive Gradient Descent'].stream(new_source_data, rollover=15)
        pos['Adaptive Gradient Descent'][0, 0] = new_source_data['x'][0]
        pos['Adaptive Gradient Descent'][1, 0] = new_source_data['y'][0]
        v['Adaptive Gradient Descent'][0, 0] += l_grad[0, 0] ** 2
        v['Adaptive Gradient Descent'][1, 0] += l_grad[1, 0] ** 2
        
        # RMSprop
        p['RMSprop'].title.text = f'RMSprop (η={eta:.1f}, β={beta:.1f}, epoch={epoch})'
        l, l_grad = gradient(x, pos['RMSprop'], y)
        v['RMSprop'][0, 0] = beta * v['RMSprop'][0, 0] + (1 - beta) * l_grad[0, 0] ** 2
        v['RMSprop'][1, 0] = beta * v['RMSprop'][1, 0] + (1 - beta) * l_grad[1, 0] ** 2
        new_source_data = dict(x = [source['RMSprop'].data['x'][-1] - eta / np.sqrt(epsilon + v['RMSprop'][0, 0] ** 2) * l_grad[0, 0]], 
                               y = [source['RMSprop'].data['y'][-1] - eta / np.sqrt(epsilon + v['RMSprop'][1, 0] ** 2) * l_grad[1, 0]])
        source['RMSprop'].stream(new_source_data, rollover=15)
        pos['RMSprop'][0, 0] = new_source_data['x'][0]
        pos['RMSprop'][1, 0] = new_source_data['y'][0]
        
        # Adam
        p['Adam'].title.text = f'Adam (η={eta:.1f}, β1={beta:.2f}, β2={beta2:.2f}, epoch={epoch})'
        l, l_grad = gradient(x, pos['Adam'], y)
        m['Adam'][0, 0] = beta * m['Adam'][0, 0] + (1 - beta) * l_grad[0, 0]
        m['Adam'][1, 0] = beta * m['Adam'][1, 0] + (1 - beta) * l_grad[1, 0]
        v['Adam'][0, 0] = beta2 * v['Adam'][0, 0] + (1 - beta2) * l_grad[0, 0] ** 2
        v['Adam'][1, 0] = beta2 * v['Adam'][1, 0] + (1 - beta2) * l_grad[1, 0] ** 2
        new_source_data = dict(x = [source['Adam'].data['x'][-1] - eta * m['Adam'][0, 0] / (1 - beta) / np.sqrt(epsilon + v['Adam'][0, 0] / (1 - beta2))], 
                               y = [source['Adam'].data['y'][-1] - eta * m['Adam'][1, 0] / (1 - beta) / np.sqrt(epsilon + v['Adam'][1, 0] / (1 - beta2))])
        source['Adam'].stream(new_source_data, rollover=15)
        pos['Adam'][0, 0] = new_source_data['x'][0]
        pos['Adam'][1, 0] = new_source_data['y'][0]
        
        # Nadam
        p['Nadam'].title.text = f'Nadam (η={eta:.1f}, β1={beta:.2f}, β2={beta2:.2f}, epoch={epoch})'
        l, l_grad = gradient(x, pos['Nadam'], y)
        grad_hat = (1 - beta) * l_grad / (1 - beta ** epoch)
        m['Nadam'][0, 0] = beta * m['Nadam'][0, 0] + (1 - beta) * l_grad[0, 0]
        m['Nadam'][1, 0] = beta * m['Nadam'][1, 0] + (1 - beta) * l_grad[1, 0]
        v['Nadam'][0, 0] = beta2 * v['Nadam'][0, 0] + (1 - beta2) * l_grad[0, 0] ** 2
        v['Nadam'][1, 0] = beta2 * v['Nadam'][1, 0] + (1 - beta2) * l_grad[1, 0] ** 2
        new_source_data = dict(x = [source['Nadam'].data['x'][-1] - eta * (m['Nadam'][0, 0] / (1 - beta ** epoch) + grad_hat[0, 0]) / np.sqrt(epsilon + v['Nadam'][0, 0] / (1 - beta2))], 
                               y = [source['Nadam'].data['y'][-1] - eta * (m['Nadam'][1, 0] / (1 - beta ** epoch) + grad_hat[1, 0]) / np.sqrt(epsilon + v['Nadam'][1, 0] / (1 - beta2))])
        source['Nadam'].stream(new_source_data, rollover=15)
        pos['Nadam'][0, 0] = new_source_data['x'][0]
        pos['Nadam'][1, 0] = new_source_data['y'][0]
        
        # Stochastic Gradient Descent
        for s in range(x.shape[0]):
            l, l_grad = gradient(x[s].reshape(1, 2), pos['Stochastic Gradient Descent'], y[s])
            p['Stochastic Gradient Descent'].title.text = f'Stochastic Gradient Descent (η={eta:.1f}, epoch={epoch})'
            new_source_data = dict(x = [source['Stochastic Gradient Descent'].data['x'][-1] - eta * l_grad[0, 0]], y = [source['Stochastic Gradient Descent'].data['y'][-1] - eta * l_grad[1, 0]])
            source['Stochastic Gradient Descent'].stream(new_source_data, rollover=45)
            pos['Stochastic Gradient Descent'][0, 0] -= eta * l_grad[0, 0]
            pos['Stochastic Gradient Descent'][1, 0] -= eta * l_grad[1, 0]
        
        # Mini-batch Gradient Descent
        index = np.arange(x.shape[0])
        np.random.shuffle(index)
        for s in np.array(np.split(index, 3)):
            l, l_grad = gradient(x[s].reshape(5, 2), pos['Mini-batch Gradient Descent'], y[s])
            p['Mini-batch Gradient Descent'].title.text = f'Mini-batch Gradient Descent (batch=5, η={eta:.1f}, epoch={epoch})'
            new_source_data = dict(x = [source['Mini-batch Gradient Descent'].data['x'][-1] - eta * l_grad[0, 0]], y = [source['Mini-batch Gradient Descent'].data['y'][-1] - eta * l_grad[1, 0]])
            source['Mini-batch Gradient Descent'].stream(new_source_data, rollover=15)
            pos['Mini-batch Gradient Descent'][0, 0] -= eta * l_grad[0, 0]
            pos['Mini-batch Gradient Descent'][1, 0] -= eta * l_grad[1, 0]
            
    elif not run.active:
        run.label = 'Start'
    elif epoch > num_epoch:
        refresh.active = True
    
    if refresh.active:
        epoch = 0
        refresh.active = False
        for method in ['Batch Gradient Descent', 'Stochastic Gradient Descent', 'Mini-batch Gradient Descent', 'Momemtum', 'Adaptive Gradient Descent', 'RMSprop', 'Adam', 'Nesterov Accelerate Gradient', 'Nadam']:
            pos[method] = np.array([x_init, y_init]).reshape(2, 1)
            new_source_data = dict(x = [copy(pos[method][0, 0])], y = [copy(pos[method][1, 0])])
            source[method].stream(new_source_data, rollover=1)
            if method in {'Momemtum', 'Adaptive Gradient Descent', 'RMSprop', 'Adam', 'Nadam'}:
                v[method] = np.array([0., 0.]).reshape(2, 1)

            if method == 'Momemtum':
                p['Momemtum'].title.text = f'Momemtum (η={eta:.1f}, 𝛾={gamma:.1f}, epoch={epoch})'
            elif method == 'Adaptive Gradient Descent':
                p['Adaptive Gradient Descent'].title.text = f'Adaptive Gradient Descent (η={eta:.1f}, epoch={epoch})'
            elif method == 'RMSprop':
                p['RMSprop'].title.text = f'RMSprop (η={eta:.1f}, β={beta:.2f}, epoch={epoch})'
            elif method == 'Adam':
                p['Adam'].title.text = f'Adam (η={eta:.1f}, β1={beta:.2f}, β2={beta2:.2f}, epoch={epoch})'
                m['Adam'] = np.array([0., 0.]).reshape(2, 1)
            elif method == 'Nesterov Accelerate Gradient':
                v['Nesterov Accelerate Gradient'] = np.array([0., 0., 0., 0.]).reshape(2, 2)
            elif method == 'Nadam':
                p['Nadam'].title.text = f'Nadam (η={eta:.1f}, β1={beta:.2f}, β2={beta2:.2f}, epoch={epoch})'
                m['Nadam'] = np.array([0., 0.]).reshape(2, 1)


# In[6]:


n = 15
x = np.array([np.random.normal(0, 0.5, n), np.random.normal(1.25, 0.5, n)]).T
y = np.matmul(x, np.array([1.56, -1.62]).T)

w0, w1 = np.meshgrid(np.linspace(-75., 75., num=400), np.linspace(-75., 75., num=400))
w = np.array([w0.ravel(), w1.ravel()])

loss = np.sqrt(np.average(((y.reshape(n, 1) - np.matmul(x, w)) ** 2), axis=0))

p, v, m, pos, source = dict(), dict(), dict(), dict(), dict()

color_mapper = LinearColorMapper(linear_palette(colorcet.bmy, 256), low=np.min(loss), high=np.max(loss))
color_bar = ColorBar(color_mapper=color_mapper, border_line_color=None, location=(0,0))


# In[7]:


eta, gamma, epsilon, beta, beta2, num_epoch = 3, 0.3, 1, 0.9, 0.99, 100
x_init, y_init = -40., -40.
epoch = 0

for method in ['Batch Gradient Descent', 'Stochastic Gradient Descent', 'Mini-batch Gradient Descent', 'Momemtum', 'Adaptive Gradient Descent', 'RMSprop', 'Adam', 'Nesterov Accelerate Gradient', 'Nadam']:
    p[method] = figure(title=f'{method} (η={eta:.1f}, epoch={epoch})', tooltips=[('w1', '$x'), ('w2', '$y'), ('loss', '@image')], tools=['pan', 'wheel_zoom', 'reset', 'save'], x_axis_label='w1', y_axis_label='w2')
    p[method].image(image=[loss.reshape(400, 400)], x=-75, y=-75, dw=150, dh=150, color_mapper=color_mapper)
    p[method].x_range.range_padding = p[method].y_range.range_padding = 0
    p[method].add_layout(color_bar, 'right')
    p[method].title.align = 'center'
    p[method].title.text_font_size = p[method].xaxis.axis_label_text_font_size = p[method].yaxis.axis_label_text_font_size = "12pt"
    
    pos[method] = np.array([x_init, y_init]).reshape(2, 1)
    source[method] = ColumnDataSource(dict(x = [copy(pos[method][0, 0])], y = [copy(pos[method][1, 0])]))

    p[method].circle(x='x', y='y', source=source[method], line_color='#ffffff', size=8, color='#ffffff')
    p[method].line(x='x', y='y', source=source[method], line_color='#ffffff', line_width=4, line_alpha=0.8)
    
    if method in {'Momemtum', 'Adaptive Gradient Descent', 'RMSprop', 'Adam', 'Nadam'}:
        v[method] = np.array([0., 0.]).reshape(2, 1)
        
    if method == 'Momemtum':
        p['Momemtum'].title.text = f'Momemtum (η={eta:.1f}, 𝛾={gamma:.1f}, epoch={epoch})'
    elif method == 'Adaptive Gradient Descent':
        p['Adaptive Gradient Descent'].title.text = f'Adaptive Gradient Descent (η={eta:.1f}, epoch={epoch})'
    elif method == 'RMSprop':
        p['RMSprop'].title.text = f'RMSprop (η={eta:.1f}, β={beta:.2f}, epoch={epoch})'
    elif method == 'Adam':
        p['Adam'].title.text = f'Adam (η={eta:.1f}, β1={beta:.2f}, β2={beta2:.2f}, epoch={epoch})'
        m['Adam'] = np.array([0., 0.]).reshape(2, 1)
    elif method == 'Nesterov Accelerate Gradient':
        v['Nesterov Accelerate Gradient'] = np.array([0., 0., 0., 0.]).reshape(2, 2)
    elif method == 'Nadam':
        p['Nadam'].title.text = f'Nadam (η={eta:.1f}, β1={beta:.2f}, β2={beta2:.2f}, epoch={epoch})'
        m['Nadam'] = np.array([0., 0.]).reshape(2, 1)


# In[8]:


run = Toggle(label="Pause", active=True, button_type="success")
refresh = Toggle(label="Refresh", active=False, button_type="success")
eta_slider = Slider(start=1, end=16, value=3, step=0.5, title="η")
gamma_slider = Slider(start=0.1, end=1.0, value=0.3, step=0.1, title="𝛾")
epsilon_slider = Slider(start=1, end=32, value=3, step=0.5, title="ε")
beta_slider = Slider(start=0.1, end=0.99, value=0.9, step=0.01, title="β")
beta2_slider = Slider(start=0.1, end=0.99, value=0.99, step=0.01, title="β2")
x_slider = Slider(start=-50., end=50., value=-40., step=1., title='x')
y_slider = Slider(start=-50., end=50., value=-40., step=1., title='y')

eta_slider.on_change('value', update_eta)
gamma_slider.on_change('value', update_gamma)
epsilon_slider.on_change('value', update_epsilon)
beta_slider.on_change('value', update_beta)
beta2_slider.on_change('value', update_beta2)
x_slider.on_change('value', update_x)
y_slider.on_change('value', update_y)

curdoc().add_root(column(row(p['Batch Gradient Descent'], p['Stochastic Gradient Descent'], p['Mini-batch Gradient Descent']), 
                         row(p['Adaptive Gradient Descent'], p['RMSprop'], p['Momemtum']),
                         row(p['Nesterov Accelerate Gradient'], p['Adam'], p['Nadam']),
                         row(run, refresh), eta_slider, gamma_slider, beta_slider, beta2_slider, x_slider, y_slider))
curdoc().add_periodic_callback(update, 500)

