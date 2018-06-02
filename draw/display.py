import matplotlib.pyplot as plt
import types, platform

def show(fig):
    """
    Creates a dummy figure and use its manager to display 'fig'
    http://stackoverflow.com/questions/31729948/matplotlib-how-to-show-a-figure-that-has-been-closed
    """
    # fig.show2 = types.MethodType(fig.show, fig)
    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)
    if hasattr(fig, 'show2'):
        fig.show2()
    else:
        fig.show()

def modify_show(fig):
    """
    Creates a copy of method 'fig.show' (fig.show2) and modify 'fig.show' in order to allow
    showing a figure that has been closed before
    http://stackoverflow.com/questions/972/adding-a-method-to-an-existing-object-instance
    """
    if platform.system == 'Windows':
        if fig:
            fig.show2 = types.MethodType(fig.show, fig)
            fig.show = types.MethodType(show, fig)
    return fig

def set_tight_plt():
    plt.gcf().subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.gca().margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

