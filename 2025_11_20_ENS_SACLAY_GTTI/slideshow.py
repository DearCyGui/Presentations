from contextlib import redirect_stdout
import dearcygui as dcg
import imageio
import io
import math
import numpy as np
import os
import threading
import time

from typing import overload, cast


# Global context storage
class _ThreadLocalList(threading.local):
    def __init__(self):
        self.data = []

    def append(self, item):
        self.data.append(item)

    def pop(self):
        return self.data.pop()

    def peek(self):
        if not self.data:
            raise RuntimeError("No context available.")
        return self.data[-1]

_global_context = _ThreadLocalList()

def _push_context(context: dcg.Context):
    """Push a new context onto the context stack."""
    global _global_context
    _global_context.append(context)

def _pop_context():
    """Pop the current context from the context stack."""
    global _global_context
    if _global_context:
        _global_context.pop()
    else:
        raise RuntimeError("No context to pop from the stack.")

def _get_context() -> dcg.Context:
    """Get the current DearCyGui context."""
    global _global_context
    if not _global_context:
        raise RuntimeError("No DearCyGui context available. Call set_context(C) first.")
    return _global_context.peek()


class _ContextManagerMeta(type):
    """
    Metaclass that makes classes usable as context managers without instantiation.
    
    This allows clean syntax like:
        with Column:
            ...
    
    Instead of requiring:
        with Column(context):
            ...
    
    When used without parentheses, automatically gets context from _get_context().
    Uses a stack-based approach to support nested context managers of the same type.
    """
    
    def __init__(cls, name, bases, dct):
        """Initialize each class with its own instance stack."""
        super().__init__(name, bases, dct)
        cls._instance_stack = []
    
    def __enter__(cls):
        """Allow using the class directly as a context manager with default parameters."""
        # Get context from stack and pass it to __init__
        context = _get_context()
        instance = cls(context)
        instance.__enter__()
        # Use a stack to support nested context managers of the same type
        cls._instance_stack.append(instance)
        return instance
    
    def __exit__(cls, exc_type, exc_val, exc_tb):
        """Exit the context manager, popping from the stack to handle nesting."""
        if cls._instance_stack:
            instance = cls._instance_stack.pop()
            return instance.__exit__(exc_type, exc_val, exc_tb)
        return False

class _ContextManagerBase(metaclass=_ContextManagerMeta):
    """Base class for context managers that build DearCyGui items.
    
    All subclasses must accept context as the first parameter in __init__.
    """
    def __init__(self, context: dcg.Context, **kwargs):
        self.context = context
        self._parent_context = None
        self._created_item = None
        # Warn if any unexpected kwargs remain
        if kwargs:
            import warnings
            warnings.warn(f"Unexpected keyword arguments: {', '.join(kwargs.keys())}", UserWarning)

    def __enter__(self):
        # Store the context
        self._parent_context = self.context
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def _get_applicable_context(*args, **kwargs) -> dcg.Context:
    """Determine the appropriate DearCyGui context from args or kwargs.
    
    If a dcg.Context is provided as the first positional argument, it is used.
    Otherwise, if a 'context' keyword argument is provided, it is used.
    If neither is provided, the current global context is returned.
    """
    if len(args) >= 1 and isinstance(args[0], dcg.Context):
        return args[0]
    elif 'context' in kwargs:
        return kwargs['context']
    else:
        return _get_context()

def _parse_positional_arg_with_optional_context(name: str, *args, **kwargs):
    """Handle argument parsing for items that accept:
    
    Item(context, value, ...)
    Item(value, ...)
    Item(..., value=..., context=C, ...)
    Item(context, ..., value=...)
    Item(..., value=...)
    """
    context = _get_applicable_context(*args, **kwargs)
    if len(args) == 2:
        if isinstance(args[0], dcg.Context):
            # Item(C, value, ...)
            value = args[1]
        else:
            raise TypeError(f"Expected (context, {name}, ...) or ({name}, ...), got ({type(args[0]).__name__}, {type(args[1]).__name__})")
        # Item(C, value, ...)
        return context, value
    elif len(args) == 1:
        if isinstance(args[0], dcg.Context):
            if 'name' in kwargs:
                # Item(C, ..., name=...)
                return context, kwargs.pop(name)
            raise TypeError(f"Missing required argument: '{name}'")
        else:
            # Item(value, ...)
            return context, args[0]
    elif 'name' in kwargs:
        # Item(..., name=..., context=C, ...)
        return context, kwargs.pop(name)
    elif len(args) == 0:
        raise TypeError(f"Missing required argument: '{name}'")
    else:
        raise TypeError(f"Received too many positional arguments")

def _parse_optional_arg_with_optional_context(name: str, default, *args, **kwargs):
    """Same as _parse_positional_arg_with_optional_context, but accepts a default value.
    
    Item(context, value, ...)
    Item(value, ...)
    Item(..., value=..., context=C, ...)
    Item(context, ..., value=...)
    Item(..., value=...)
    Item(context, ...)  # uses default
    Item(...)  # uses default
    """
    context = _get_applicable_context(*args, **kwargs)
    if len(args) == 2:
        if isinstance(args[0], dcg.Context):
            # Item(C, value, ...)
            value = args[1]
        else:
            raise TypeError(f"Expected (context, {name}, ...) or ({name}, ...), got ({type(args[0]).__name__}, {type(args[1]).__name__})")
        return context, value
    elif len(args) == 1:
        if isinstance(args[0], dcg.Context):
            # Item(C, ..., name=...) or Item(C, ...) with default
            if name in kwargs:
                return context, kwargs.pop(name)
            else:
                return context, default
        else:
            # Item(value, ...)
            return context, args[0]
    elif name in kwargs:
        # Item(..., name=..., context=C, ...)
        return context, kwargs.pop(name)
    elif len(args) == 0:
        # Item(...) with default
        return context, default
    else:
        raise TypeError(f"Received too many positional arguments")


def _extract_auto_resize_x(requested_width: float | dcg.baseSizing | None):
    """Handle parsing an optional width attribute where the default (None) means auto-resize.
    
    floating point values <= 1 are treated as fractions of available width.
    """
    if requested_width is None:
        # Default behavior: auto-resize to content
        width = 0
        auto_resize_x = True
    else:
        auto_resize_x = False
        # Fixed width as fraction of parent width
        if isinstance(requested_width, dcg.baseSizing):
            width = requested_width
        else:
            if requested_width <= 0:
                raise ValueError("Column width must be positive or None")
            if requested_width <= 1: # Percentage of available width
                width = dcg.Size.FULLX() * requested_width
            else:
                width = requested_width * dcg.Size.DPI()
    return auto_resize_x, width

def _extraction_auto_resize_y(requested_height: float | dcg.baseSizing | None):
    """Handle parsing an optional height attribute where the default (None) means auto-resize.
    
    floating point values <= 1 are treated as fractions of available height.
    """
    if requested_height is None:
        # Default behavior: auto-resize to content
        height = 0
        auto_resize_y = True
    else:
        auto_resize_y = False
        # Fixed height as fraction of parent height
        if isinstance(requested_height, dcg.baseSizing):
            height = requested_height
        else:
            if requested_height <= 0:
                raise ValueError("Column height must be positive or None")
            if requested_height <= 1: # Percentage of available height
                height = dcg.Size.FULLY() * requested_height
            else:
                height = requested_height * dcg.Size.DPI()
    return auto_resize_y, height


def _get_unique_filename(base_name: str, ext: str) -> str:
    """Generate a unique filename by appending a number if file exists"""
    if not os.path.exists(f"{base_name}.{ext}"):
        return f"{base_name}.{ext}"
    counter = 1
    while os.path.exists(f"{base_name}_({counter}).{ext}"):
        counter += 1
    return f"{base_name}_({counter}).{ext}"

def export_slideshow(target : str, slideshow : 'SlideShow'):
    """Export the slideshow to a pdf file.
    """
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.utils import ImageReader
    except ImportError:
        raise ImportError("Exporting slides requires reportlab")
    images = []
    slideshow.context.viewport.retrieve_framebuffer = True
    current_slide_num = slideshow.current_slide
    slideshow.start()
    for _ in range(len(slideshow.slides)):
        # force convergence of the content
        # ChildWindows can take a few frames to converge,
        # and we need center callbacks to converge as well
        try:
            slideshow.context.viewport.render_frame()
            slideshow.context.viewport.wake()
            slideshow.context.viewport.render_frame()
            slideshow.context.viewport.wake()
            time.sleep(0.1)
            slideshow.context.viewport.render_frame()
            slideshow.context.viewport.wake()
        except:
            # Not run from main thread. We'll assume there is a render_frame loop.
            slideshow.context.viewport.wake()
            time.sleep(0.1)
            slideshow.context.viewport.wake()
            pass
        slideshow.next_slide()
        images.append(slideshow.context.viewport.framebuffer.read())
    slideshow.context.viewport.retrieve_framebuffer = False
    slideshow.show_slide(current_slide_num)

    # Create PDF with reportlab
    if len(images) > 0:
        # Get size from first image
        img_width = images[0].shape[1]
        img_height = images[0].shape[0]
        
        # Create PDF canvas with custom page size (converting pixels to points)
        points_width = img_width * 72 / 96  # 72 points per inch, assuming 96 DPI
        points_height = img_height * 72 / 96
        
        c = canvas.Canvas(target, pagesize=(points_width, points_height))
        
        for img_array in images: 
            # Convert RGBA to RGB and flip vertically
            rgb_array = img_array[::-1, :, :3]
            
            # Create a virtual "file" in memory
            img_data = io.BytesIO()
            
            # Save numpy array as PNG directly to memory
            imageio.v3.imwrite(img_data, rgb_array, extension=".png")
            img_data.seek(0)
            
            # Draw image on the page
            c.drawImage(ImageReader(img_data), 0, 0, width=points_width, height=points_height)
            c.showPage()
        
        c.save()
    else:
        raise ValueError("No slides to export")

#################

class EmbeddedImage(dcg.Plot):
    """A sophisticated image display widget built on dcg.Plot.
    
    This widget provides enhanced image display capabilities compared to the basic
    dcg.Image widget, including proper aspect ratio handling and zoom capabilities.
    
    Args:
        texture (dcg.Texture, optional): Initial texture to display
        **kwargs: Additional arguments passed to dcg.Plot
    
    Features:
        - Maintains proper aspect ratio
        - Supports zooming and panning
        - Right-click to open in fullscreen popup
        - Clean minimal interface without axes or controls
    
    Example:
        ```python
        # Create from numpy array
        img_array = np.zeros((100, 100, 3), dtype=np.uint8)
        texture = dcg.Texture(C, img_array)
        image_display = EmbeddedImage(C, texture=texture)
        
        # Update image
        new_texture = dcg.Texture(C, new_img_array)
        image_display.texture = new_texture
        ```
    """
    @overload
    def __init__(self, context: dcg.Context, content: dcg.Texture, **kwargs): ...

    @overload
    def __init__(self, context: dcg.Context, content: str, **kwargs): ...

    @overload
    def __init__(self, context: dcg.Context, content: 'dcg.Array', **kwargs): ...

    @overload
    def __init__(self, content: dcg.Texture, **kwargs): ...

    @overload
    def __init__(self, content: str, **kwargs): ...

    @overload
    def __init__(self, content: 'dcg.Array', **kwargs): ...

    @overload
    def __init__(self, context: dcg.Context, **kwargs): ...

    @overload
    def __init__(self, **kwargs): ...

    def __init__(self, *args,
                 **kwargs):
        # Handle all signatures
        context, content = _parse_optional_arg_with_optional_context("content", None, *args, **kwargs)

        super().__init__(context, **kwargs)

        # Configure plot to be minimal
        self.no_mouse_pos = True
        self.no_menus = True
        self.no_frame = True
        self.no_title = True
        self.no_legend = True
        self.equal_aspects = True
        self.theme = dcg.ThemeStyleImPlot(context, plot_border_size=0)
        
        # Setup axes with no visual elements
        x_axis = self.X1
        x_axis.no_gridlines = True
        x_axis.no_tick_marks = True
        x_axis.no_tick_labels = True
        x_axis.no_side_switch = True

        y_axis = self.Y1
        y_axis.no_gridlines = True
        y_axis.no_tick_marks = True 
        y_axis.no_tick_labels = True
        y_axis.no_side_switch = True
        y_axis.invert = True

        # Create image series
        with dcg.DrawInPlot(context, parent=self):
            self._image = dcg.DrawImage(context, pmin=(0, 0), pmax=(1, 1))
        
        if isinstance(content, dcg.Texture):
            self.texture = content
        elif isinstance(content, str):
            array = imageio.v3.imread(content)
            self.texture = dcg.Texture(context, array)
        elif content is not None:
            self.texture = dcg.Texture(context, content)

        self.handlers += [
            dcg.ClickedHandler(context, callback=self._on_right_click, button=dcg.MouseButton.RIGHT),
        ]

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, _get_applicable_context(*args, **kwargs))

    @property
    def texture(self):
        return self._image.texture
        
    @texture.setter 
    def texture(self, value : dcg.Texture):
        self._image.texture = value
        self._image.pmax = (value.width, value.height)
        self.X1.fit()
        self.Y1.fit()

    def _on_right_click(self):
        """Show an enlarged version of the image in a popup"""
        if self.texture is None:
            return
        if type(self.parent) == dcg.Window and \
           self.parent.modal and \
           self.width == -1 and self.height == -1:
            self.parent.detach_item()
            return # In popup. Close it
        with dcg.Window(self.context, modal=True,
                        x=0, y=0,
                        width=self.context.viewport.width,
                        height=self.context.viewport.height):
            EmbeddedImage(self.context, texture=self.texture, width=dcg.Size.FILLX(), height=dcg.Size.FILLY())

class InteractiveCode(dcg.ChildWindow):
    """An interactive code editor with live output display.
    
    This widget provides a text editor for Python code along with output display capabilities.
    It can show text output, plots, and images generated by the executed code.
    
    Args:
        initial_code (str): Initial code to show in the editor
        width (int): Width of the editor widget (default: 440)
        height (int): Height of the editor widget (default: 440)
        **kwargs: Additional arguments passed to dcg.ChildWindow
    
    Features:
        - Syntax highlighting for Python code
        - Live code execution with "Run" button
        - Support for displaying:
            - Text output (print statements)
            - Images (numpy arrays)
            - Plots (x,y coordinate pairs)
        - Error handling and status display
    
    Example:
        ```python
        code_editor = InteractiveCode(C, initial_code="print('Hello World!')")
        code_editor.display_result()  # Creates output display widgets
        ```
    
    Note:
        The code is executed in a restricted namespace with access to numpy (as np)
        for safety. Additional modules must be imported within the code.
    """
    @overload
    def __init__(self, context: dcg.Context, code: str = "",
                 width: int = 440, height: int = 440, **kwargs): ...
    
    @overload
    def __init__(self, code: str = "",
                 width: int = 440, height: int = 440, **kwargs): ...
    
    def __init__(self, *args,
                 width=440, height=440,
                 **kwargs):
        # Handle both signatures
        context, initial_code = _parse_positional_arg_with_optional_context("code", *args, **kwargs)
        super().__init__(context, width=width, height=height, **kwargs)
        self.no_scrollbar = True
        self.border = False
        self.no_scroll_with_mouse = True
        
        # Add code editor
        self.editor = dcg.InputText(context, parent=self,
            width=dcg.Size.FILLX(),
            height=dcg.Size.FILLY() - dcg.Size.FIXED(34) * dcg.Size.DPI(),
            multiline=True,
            tab_input=True,
            font=dcg.AutoFont.get_monospaced(context),
            value=initial_code)
        with dcg.HorizontalLayout(context, parent=self,
                                  alignment_mode=dcg.Alignment.RIGHT):
            dcg.Button(context, label="Run", callback=self._run_code)
            self.status = dcg.Text(context, value="")

        self._last_log = ""
        self._last_output = None
        self.output_text = None
        self.image_display = None
        self.plot_display = None

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, _get_applicable_context(*args, **kwargs))

    def _run_code(self):
        """Execute the code and capture output"""
        try:
            # Create clean namespace
            namespace = {'__name__': '__main__',
                        'np': np}  # Always include numpy
            
            # Capture stdout
            with redirect_stdout(io.StringIO()) as stdout:
                wrapped_code = f"""
def _exec_with_return():
{(''.join('    ' + line + '\n' for line in self.editor.value.splitlines()))}
"""
                exec(wrapped_code, namespace)
                result = namespace['_exec_with_return']()
            
            self._last_log = stdout.getvalue()
            self._last_output = result
            self.status.value = "Success"
            self.status.color = (50, 205, 50)
            
        except Exception as e:
            self._last_log = f"Error: {str(e)}"
            self._last_output = None
            self.status.value = "Error"
            self.status.color = (220, 50, 50)
        
        self._update_result()

    def _update_result(self):
        """Update the content of output widgets"""
        if self.output_text is None:
            return

        if self._last_output is None:
            self.output_text.value = self._last_log
            if self.image_display:
                self.image_display.show = False
            if self.plot_display:
                self.plot_display.show = False
            return

        # Handle image output
        if isinstance(self._last_output, np.ndarray) and self.image_display:
            self.image_display.texture = dcg.Texture(self.context, self._last_output)
            image_text = "Image: [{}]".format(", ".join(map(str, self._last_output.shape)))
            self.output_text.value = f"{self._last_log}\n{image_text}" if self._last_log else image_text
            self.image_display.show = True
            if self.plot_display:
                self.plot_display.show = False
            
        # Handle plot output
        elif isinstance(self._last_output, tuple) and len(self._last_output) == 2 and self.plot_display:
            self.plot_display.children = []
            if not(isinstance(self._last_output[0], tuple)) and not(isinstance(self._last_output[1], tuple)):
                x = self._last_output[0]
                y = self._last_output[1]
                dcg.PlotLine(self.context, X=x, Y=y, no_legend=True, parent=self.plot_display)
            elif isinstance(self._last_output[1], tuple):
                x = self._last_output[0]
                for y in self._last_output[1]:
                    dcg.PlotLine(self.context, X=x, Y=y, no_legend=True, parent=self.plot_display)
            elif isinstance(self._last_output[0], tuple) == 1:
                y = self._last_output[1]
                for x in self._last_output[0]:
                    dcg.PlotLine(self.context, X=x, Y=y, no_legend=True, parent=self.plot_display)
            else:
                for x, y in zip(self._last_output[0], self._last_output[1]):
                    dcg.PlotLine(self.context, X=x, Y=y, no_legend=True, parent=self.plot_display)
            
            self.plot_display.X1.fit()
            self.plot_display.Y1.fit()
            self.output_text.value = self._last_log
            self.plot_display.show = True
            if self.image_display:
                self.image_display.show = False
            
        # Handle other output
        else:
            self.output_text.value = f"{self._last_log}\n{self._last_output}"
            if self.image_display:
                self.image_display.show = False
            if self.plot_display:
                self.plot_display.show = False

    def display_result(self, **kwargs):
        """Create output widgets"""
        self.output_text = dcg.Text(self.context, value="", **kwargs)
        self.image_display = EmbeddedImage(self.context, width=-1, height=300, show=False)
        self.plot_display = dcg.Plot(self.context, width=-1, height=300, 
                                   label="Interactive Plot", show=False, **kwargs)

class Center(_ContextManagerBase):
    def __init__(self, context: dcg.Context):
        super().__init__(context)

    def __enter__(self):
        container = \
            dcg.ChildWindow(
                self.context,
                border=False,
                auto_resize_x=True,
                auto_resize_y=True,
                always_auto_resize=True,
                no_scroll_with_mouse=True,
                no_scrollbar=True,
                x=dcg.Size.PARENT_XC()-dcg.Size.SELF_WIDTH()*0.5,
                y=dcg.Size.PARENT_YC()-dcg.Size.SELF_HEIGHT()*0.5)
        self.context.push_next_parent(container)
        return container

    def __exit__(self, a, b, c):
        container = self.context.fetch_parent_queue_back()
        self.context.pop_next_parent()
        assert isinstance(container, dcg.ChildWindow)
        return False

class CenterH(_ContextManagerBase):
    @overload
    def __init__(self, context: dcg.Context, width: float | dcg.baseSizing | None = None): ...
    
    @overload
    def __init__(self, width: float | dcg.baseSizing | None = None): ...
    
    def __init__(self, *args, **kwargs):
        context, width = _parse_optional_arg_with_optional_context("width", None, *args, **kwargs)
        super().__init__(context)
        self._width = width

    def __enter__(self):
        auto_resize_x, width = _extract_auto_resize_x(self._width)
        if auto_resize_x:
            container = \
                dcg.ChildWindow(
                    self.context,
                    border=False,
                    auto_resize_x=True,
                    auto_resize_y=True,
                    always_auto_resize=True,
                    no_scroll_with_mouse=True,
                    no_scrollbar=True,
                    x=dcg.Size.PARENT_XC()-dcg.Size.SELF_WIDTH()*0.5)
        else:
            container = \
                dcg.HorizontalLayout(
                    self.context,
                    alignment_mode=dcg.Alignment.CENTER,
                    no_wrap=False,
                    width=width)
        self.context.push_next_parent(container)
        return container

    def __exit__(self, a, b, c):
        container = self.context.fetch_parent_queue_back()
        self.context.pop_next_parent()
        assert isinstance(container, (dcg.ChildWindow, dcg.HorizontalLayout))
        return False

class CenterV(_ContextManagerBase):
    @overload
    def __init__(self, context: dcg.Context, height: float | dcg.baseSizing | None = None): ...
    
    @overload
    def __init__(self, height: float | dcg.baseSizing | None = None): ...
    
    def __init__(self, *args, **kwargs):
        context, height = _parse_optional_arg_with_optional_context("height", None, *args, **kwargs)
        super().__init__(context)
        self._height = height

    def __enter__(self):
        auto_resize_y, height = _extraction_auto_resize_y(self._height)
        container = \
            dcg.ChildWindow(
                self.context,
                border=False,
                width=dcg.Size.FILLX(),
                auto_resize_y=auto_resize_y,
                height=height,
                always_auto_resize=True,
                no_scroll_with_mouse=True,
                no_scrollbar=True,
                y=dcg.Size.PARENT_YC()-dcg.Size.SELF_HEIGHT()*0.5)
        self.context.push_next_parent(container)
        return container

    def __exit__(self, a, b, c):
        container = self.context.fetch_parent_queue_back()
        self.context.pop_next_parent()
        assert isinstance(container, dcg.ChildWindow)
        return False

class Column(_ContextManagerBase):
    @overload
    def __init__(self, context: dcg.Context, width: float | dcg.baseSizing | None = None): ...
    
    @overload
    def __init__(self, width: float | dcg.baseSizing | None = None): ...
    
    def __init__(self, *args, **kwargs):
        context, width = _parse_optional_arg_with_optional_context("width", None, *args, **kwargs)
        super().__init__(context)
        self._width = width

    def __enter__(self):
        auto_resize_x, width = _extract_auto_resize_x(self._width)
        container = \
            dcg.ChildWindow(
                self.context,
                border=False,
                width=width,
                auto_resize_x=auto_resize_x,
                auto_resize_y=True,
                always_auto_resize=True,
                no_scroll_with_mouse=True,
                no_scrollbar=True)
        self.context.push_next_parent(container)
        return container

    def __exit__(self, a, b, c):
        container = self.context.fetch_parent_queue_back()
        self.context.pop_next_parent()
        assert isinstance(container, dcg.ChildWindow)
        return False

class Columns(_ContextManagerBase):
    def __init__(self, context: dcg.Context):
        super().__init__(context)

    def __enter__(self):
        container = dcg.Layout(self.context)
        self.context.push_next_parent(container)
        return container

    def __exit__(self, a, b, c):
        container = self.context.fetch_parent_queue_back()
        self.context.pop_next_parent()
        assert isinstance(container, dcg.Layout)

        children = container.children
        if len(children) <= 1:
            return False

        for child in children:
            if not isinstance(child, (dcg.ChildWindow, dcg.Layout)):
                raise ValueError(f"Columns doesn't support {child}")

        col_separator = dcg.Size.THEME_STYLE("item_spacing", False)

        child_sizes = [child.width for child in children]

        # placement if we were left-aligned
        left_align_positions = [container.x.x1]
        for size in child_sizes[:-1]:
            left_align_positions.append(
                left_align_positions[-1] + col_separator + size
            )

        # placement if we were right-aligned
        right_align_positions = [container.x.x2 - child_sizes[-1]]
        for size in reversed(child_sizes[:-1]):
            right_align_positions.append(
                right_align_positions[-1] - col_separator - size
            )
        right_align_positions.reverse()

        # placement if we were justified
        ideal_separator = container.width.content_width / len(children) # not (len(children) - 1) because we add space to the right item as well
        justified_align_positions = [container.x.x1]
        for size in child_sizes[:-1]:
            justified_align_positions.append(
                justified_align_positions[-1] + ideal_separator + size
            )

        for i, child in enumerate(children):
            # position must be no less than the left aligned position
            # position must be no more than left of the right aligned position
            # position is ideally left-justified
            child.x = dcg.Size.MAX(
                left_align_positions[i],
                dcg.Size.MIN(
                    right_align_positions[i],
                    justified_align_positions[i]
                )
            )
            child.no_newline = True
        return False

class FootNote(_ContextManagerBase):
    def __init__(self, context: dcg.Context):
        super().__init__(context)

    def __enter__(self):
        container = \
            dcg.Layout(self.context, label="footnote")
        self.context.push_next_parent(container)
        dcg.Separator(self.context)
        return container

    def __exit__(self, a, b, c):
        container = self.context.fetch_parent_queue_back()
        self.context.pop_next_parent()
        assert isinstance(container, dcg.Layout)
        return False

def _get_parent_width_limit(container: dcg.uiItem) -> float:
    """Recursively find the nearest width limit from parent containers."""
    parent = container.parent
    if parent is None:
        return 100 # shouldn't occur
    assert not isinstance(parent, dcg.plotElement)
    if isinstance(parent, dcg.ChildWindow):
        if parent.auto_resize_x:
            return _get_parent_width_limit(parent)
    if hasattr(parent, 'state') and hasattr(parent.state, 'content_region_avail'):
        avail_rect = parent.state.content_region_avail
        return float(avail_rect[0])
    if hasattr(parent, 'state') and hasattr(parent.state, 'rect_size'):
        rect = parent.state.rect_size
        return float(rect[0])
    return _get_parent_width_limit(parent)

def _get_parent_height_limit(container: dcg.uiItem) -> float:
    """Recursively find the nearest height limit from parent containers."""
    parent = container.parent
    if parent is None:
        return 100 # shouldn't occur
    assert not isinstance(parent, dcg.plotElement)
    if isinstance(parent, dcg.ChildWindow):
        if parent.auto_resize_y:
            return _get_parent_height_limit(parent)
    if hasattr(parent, 'state') and hasattr(parent.state, 'content_region_avail'):
        avail_rect = parent.state.content_region_avail
        return float(avail_rect[1])
    if hasattr(parent, 'state') and hasattr(parent.state, 'rect_size'):
        rect = parent.state.rect_size
        return float(rect[1])
    return _get_parent_height_limit(parent)


class _RPROPScalingOptimizer:
    """iRPROP-style adaptive optimizer for scaling_factor convergence.
    
    This class implements an adaptation of the improved Resilient
    Backpropagation (iRPROP) algorithm, for the purpose of scaling factor
    optimization.

    In particular the adaptation are that:
    - For the "gradient" sign, we use the sign of the error (target_scale_factor - 1.0)
        Basically if it is negative, we need to reduce scale (overshoot), if
        positive we need to increase scale.
    - For the end result, overshooting is bad. Thus the optimizer favors non-overshooting
       (in an RPROP- like way) by reverting to the last known non-overshooting scales.
    - Parameters are tuned for this task
    - Since the "loss" is not contiguous (due to discrete layout changes, e.g.,
      from theme component rounding), we do not use error thresholds for convergence,
      but only rely on step size becoming too small.
    - Optimization is reset when the error changes in an unexpected way, indicating an external change
      such as parent resize.
    - The optimizer doesn't accept overshooting as a valid state for convergence.
    - parameter updates are done in log space (multiplicative changes to scale factor).
    
    Args:
        initial_step_size (float): Starting step size (default: 0.05 = 5%)
        step_increase (float): Multiplier when improving (default: 1.2 = 20% increase)
        step_decrease (float): Multiplier when oscillating (default: 0.2 = 80% reduction)
        min_step (float): Minimum step size (default: 0.002 = 0.2% in log space)
        max_step (float): Maximum step size (default: 1.0 = 100%)
    """
    
    def __init__(self,
                 initial_step_size: float = 0.05,
                 step_increase: float = 1.2,
                 step_decrease: float = 0.2,
                 min_step: float = 0.002,
                 max_step: float = 1.0):
        self.initial_step_size = initial_step_size
        self.step_increase = step_increase
        self.step_decrease = step_decrease
        self.min_step = min_step
        self.max_step = max_step
        
        # State variables
        self.reset()
    
    def reset(self):
        """Reset the optimizer to initial state."""
        self.prev_error = None
        self.step_size = self.initial_step_size
        self.last_non_overshoot_scale = None
        self.is_converged_state = False
        self.is_reset = True
    
    def check_convergence(self, current_error: float) -> bool:
        """Check if the optimizer has converged based on step size.
        
        Also detects dramatic error changes that indicate external changes (like parent resize).
        
        Args:
            current_error: Current error value
            
        Returns:
            True if converged (step size too small), False otherwise
        """
        # Check incoherences in error trend that indicate external changes
        if self.prev_error is not None:
            epsilon = 1e-3 # take into account small numerical noise
            if current_error < self.prev_error - epsilon and self.prev_error < 0:
                # The error is getting worse (more negative), while we are decreasing scale
                # This indicates a change of conditions (e.g., parent resize)
                self.reset()
                return False
            if current_error > self.prev_error + epsilon and self.prev_error > 0:
                # The error is getting worse (more positive), while we are increasing scale
                # This indicates a change of conditions (e.g., parent resize)
                self.reset()
                return False
            if self.is_converged_state:
                # If we were converged, but error increased, reset
                if abs(current_error) > abs(self.prev_error):
                    self.reset()
                    return False

        # Never converge when overshooting - we need to keep adjusting
        if current_error < 0.0: # overshoot
            self.prev_error = current_error
            self.is_converged_state = False
            return False
        
        # Check if step size is below threshold - this means we've converged
        step_converged = self.step_size <= self.min_step

        if self.is_converged_state and not step_converged:
            # If we were converged but not anymore, we need to reset
            self.reset()
            return False

        # Update converged state and prev_error
        self.is_converged_state = step_converged
        self.prev_error = current_error
        
        return step_converged
    
    def compute_adjustment(self, current_error: float, last_scale: float, theoretical_scale: float) -> float:
        """Compute the new scale factor using iRPROP algorithm.
        
        Args:
            current_error: Current error (target_scale_factor - 1.0)
            last_scale: Current scaling factor
            theoretical_scale: The theoretical scale factor if there were no rounding effects
            
        Returns:
            - new_scale: The new scaling factor to apply
        """
        if self.is_reset:
            # Initialize search
            self.is_reset = False
            return theoretical_scale

        error_sign = 1.0 if current_error > 0 else -1.0 if current_error < 0 else 0.0

        # Track non-overshooting scales
        is_overshoot = current_error < 0.0
        if not is_overshoot:
            self.last_non_overshoot_scale = last_scale
        
        # If we overshoot and have a previous non-overshoot scale that was lower, revert to it
        if is_overshoot and self.last_non_overshoot_scale is not None:
            if self.last_non_overshoot_scale < last_scale:
                # Revert to last non-overshoot scale and reduce step size like in oscillation
                self.step_size *= self.step_decrease
                self.step_size = max(self.step_size, self.min_step)
                # Don't reset prev_error here - keep the one of last_non_overshoot_scale
                return self.last_non_overshoot_scale
        
        # iRPROP adaptive step size adjustment based on previous error
        # Note: prev_error is updated in check_convergence, so we use the value from last iteration
        # Check if error changed sign (oscillation detected)
        if self.prev_error is not None:
            if current_error * self.prev_error < 0:
                # Oscillation: reduce step size for finer adjustment
                self.step_size *= self.step_decrease
                self.step_size = max(self.step_size, self.min_step)
            elif abs(current_error) < abs(self.prev_error):
                # Error is decreasing: increase step size slightly for faster convergence
                self.step_size *= self.step_increase
                self.step_size = min(self.step_size, self.max_step)
            # If error increased without sign change, keep step size (might be discrete jump)
        
        # Calculate adjustment using adaptive step size
        # Apply step in the direction of the error in log space
        new_scale = math.exp(math.log(last_scale) + error_sign * self.step_size)

        # Safety bounds: don't shrink below 10% or grow beyond 500%
        new_scale = max(0.1, min(5.0, new_scale))

        return new_scale


class _DiscreteScalingOptimizer:
    """Discrete scaling optimizer with fixed step sizes.
    
    This optimizer uses a fixed set of 10 scaling factors (evenly spaced in log scale)
    and performs a binary-search-like approach to find the optimal scale:
    - Overshoot: decrease scale factor (try smaller scale)
    - No overshoot: increase scale factor (try larger scale)
    - Track best non-overshooting scale
    - Reset when error changes unexpectedly
    
    Args:
        min_scale (float): Minimum scaling factor (default: 0.25)
        max_scale (float): Maximum scaling factor (default: 4.0)
        num_steps (int): Number of discrete scale steps (default: 11)
    """
    
    def __init__(self,
                 min_scale: float = 0.25,
                 max_scale: float = 4.0,
                 num_steps: int = 11):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.num_steps = num_steps
        
        # Create fixed scale factors evenly spaced in log space
        log_min = math.log(min_scale)
        log_max = math.log(max_scale)
        self.scale_factors = [
            math.exp(log_min + i * (log_max - log_min) / (num_steps - 1))
            for i in range(num_steps)
        ]
        
        # State variables
        self.reset()
    
    def reset(self):
        """Reset the optimizer to initial state."""
        self.best_abs_error = None
        self.current_index = len(self.scale_factors) // 2  # Start in middle
        self.best_non_overshoot_index = len(self.scale_factors) - 1  # Track best non-overshooting scale
        self.lowest_overshoot_index = 0  # Track lowest scale that overshoots
        self.is_reset = True
    
    def check_convergence(self, current_error: float) -> bool:
        """Check if the optimizer has converged.
        
        Converges when we've found the best non-overshooting scale
        (next higher scale overshoots).
        
        Args:
            current_error: Current error value
            
        Returns:
            True if converged, False otherwise
        """
        converged = self.lowest_overshoot_index <= self.best_non_overshoot_index + 1
        # Check incoherences in error trend that indicate external changes
        if self.best_abs_error is not None:
            if converged:
                # If we were converged but error increased, reset
                if abs(current_error) > self.best_abs_error:
                    self.reset()
                    return False
                if abs(current_error) > 0.3: # arbitrary threshold
                    self.reset()
                    return False
        
        # Never converge when overshooting
        if current_error < 0.0:
            if converged:
                self.reset()
                return False

    
        self.best_abs_error = abs(current_error) \
            if self.best_abs_error is None else min(self.best_abs_error, abs(current_error))
        return converged
    
    def compute_adjustment(self, current_error: float, last_scale: float, theoretical_scale: float) -> float:
        """Compute the new scale factor using discrete steps.
        
        Args:
            current_error: Current error (target_scale_factor - 1.0)
            last_scale: Current scaling factor
            theoretical_scale: Theoretical scale (not used in discrete version)
            
        Returns:
            The new scaling factor to apply
        """
        if self.is_reset:
            self.is_reset = False
            # Start from closest point to theorical scale
            closest_index = min(range(len(self.scale_factors)),
                                key=lambda i: abs(self.scale_factors[i] - theoretical_scale))
            self.current_index = closest_index
            return self.scale_factors[self.current_index]

        prev_index = self.current_index

        is_overshoot = current_error < 0.0
        
        if is_overshoot:
            # Track lowest scale that overshoots
            self.lowest_overshoot_index = min(self.current_index, self.lowest_overshoot_index)
        else:
            # No overshoot: update best non-overshoot
            self.best_non_overshoot_index = max(self.current_index, self.best_non_overshoot_index)

        # Fix bounds
        self.best_non_overshoot_index = max(0, min(self.lowest_overshoot_index - 1, self.best_non_overshoot_index))

        # Try increasing scale (binary search style)
        self.current_index = (self.lowest_overshoot_index + self.best_non_overshoot_index) // 2

        # Clamp to valid range
        self.current_index = max(0, min(len(self.scale_factors) - 1, self.current_index))
        
        return self.scale_factors[self.current_index]


class _FillBase(_ContextManagerBase):
    """Base class for auto-scaling containers.
    
    Provides common infrastructure for Fill, FillV, and FillH variants.
    Subclasses override _calculate_scale() to implement different scaling strategies.
    
    Uses an iRPROP-style adaptive algorithm (via _RPROPScalingOptimizer) to handle 
    oscillations caused by theme component rounding and discrete scaling effects.
    
    Args:
        fill_percent (float): Target fill percentage (0.0 to 1.0). Default is 1.0 (100%).
    """
    
    def __init__(self, context: dcg.Context, fill_percent: float):
        super().__init__(context)
        self._fill_percent = max(0.0, min(1.0, fill_percent))
    
    def __enter__(self):
        # Create the container with initial scaling
        container = dcg.VerticalLayout(
            self.context
        )
        
        # Capture current theme and font to reapply when scaling changes
        container.theme = _get_current_theme(container)
        container.font = _get_current_font(container)
        
        # Set initial scaling
        container.scaling_factor = 1.0
        
        # Create persistent handler (callback will change itself)
        handler = dcg.RenderHandler(self.context)
        container.handlers += [
            handler,
            dcg.LostRenderHandler(self.context,
                                  callback= lambda: self._set_callback(handler, 5)) # when not visible, force a wait for convergence when visible again)
        ]
        
        # Create iRPROP optimizer with default parameters
        # For now discrete scaling works better (else there are too many font variants)
        handler.user_data = _DiscreteScalingOptimizer()# _RPROPScalingOptimizer()
        
        # Set initial callback that waits for frames before adjusting
        self._set_callback(handler, frames_to_wait=4)

        self.context.push_next_parent(container)
        return container

    def __exit__(self, a, b, c):
        container = self.context.fetch_parent_queue_back()
        self.context.pop_next_parent()
        assert isinstance(container, dcg.VerticalLayout)
        return False
    
    def _set_callback(self, handler: dcg.RenderHandler, frames_to_wait: int):
        """Set a callback that waits for a number of frames before measuring."""

        if frames_to_wait <= 0:
            callback_change = handler.callback is not self._check_and_adjust_callback

            if callback_change:
                # Set callback that performs adjustment immediately
                handler.callback = self._check_and_adjust_callback
                handler.context.viewport.wake(full_refresh=False) # run callback as soon as possible
            return

        # wait the target number of frames before taking action
        def wait_callback(handler: dcg.RenderHandler, container: dcg.VerticalLayout, frames_to_wait=frames_to_wait):
            # Safety check to prevent double spawn (two frames, delay in callback processing)
            dcg_callback = cast(dcg.Callback, handler.callback)
            if dcg_callback.callback is not wait_callback:
                return  # Skip. Each callback is single-use
            
            self._set_callback(handler, frames_to_wait-1 if container.state.visible else 5)

        handler.callback = wait_callback
        handler.context.viewport.wake(full_refresh=False)

    def _calculate_scale(self, scale_x: float, scale_y: float) -> float:
        """Calculate the new scale factor. Override in subclasses.
        
        Args:
            scale_x: The scale factor needed to fill width
            scale_y: The scale factor needed to fill height
            
        Returns:
            The new scale factor to apply
        """
        raise NotImplementedError("Subclasses must implement _calculate_scale")

    def _check_and_adjust_callback(self, handler: dcg.RenderHandler, container: dcg.VerticalLayout) -> None:
        """
        Dynamically adjust the scaling factor to fill the parent area using iRPROP algorithm.
        
        Uses the _RPROPScalingOptimizer to adaptively converge to the target scale,
        handling oscillations from discrete layout changes.
        """

        if container.parent is None:
            return
        
        # Get parent dimensions (available space)
        parent_width = _get_parent_width_limit(container)
        parent_height = _get_parent_height_limit(container)

        if parent_width <= 0 or parent_height <= 0:
            return

        if not container.state.visible or container.parent.state.resized:
            # wait convergence when visible again
            self._set_callback(handler, 5)
            return
        
        # Get container's current size
        if not hasattr(container.state, 'rect_size'):
            return
        
        container_rect = container.state.rect_size
        content_width = float(container_rect[0])
        content_height = float(container_rect[1])
        
        if content_width <= 0 or content_height <= 0:
            return

        # Capture changes to theme or font from time to time
        if container.context.viewport.metrics.frame_count % 30 == 0:
            container.theme = _get_current_theme(container)
            container.font = _get_current_font(container)

        # Get optimizer from handler (stored directly)
        optimizer = handler.user_data
        
        # Target dimensions (parent size * fill percentage)
        target_width = parent_width * self._fill_percent
        target_height = parent_height * self._fill_percent
        
        # Calculate scale factors needed for each dimension
        scale_x = target_width / content_width
        scale_y = target_height / content_height
        
        # Calculate new scale using subclass-specific strategy
        new_scale_factor = self._calculate_scale(scale_x, scale_y)

        # Calculate error: positive means we need to scale up, negative means scale down (overshoot)
        current_error = new_scale_factor - 1.0
        
        # Calculate theoretical scale: what the scale should be if applied to current content
        theoretical_scale = new_scale_factor * container.scaling_factor
        
        # Check convergence (also handles reset if error increases while converged)
        if optimizer.check_convergence(current_error):
            return
        
        # Compute new scale using iRPROP algorithm
        last_scale = container.scaling_factor
        new_scale = optimizer.compute_adjustment(current_error, last_scale, theoretical_scale)

        if new_scale == last_scale:
            # No change
            return

        # Apply the new scale
        container.scaling_factor = new_scale

        # Force a re-render
        container.context.viewport.wake(full_refresh=False)
        
        # wait for convergence over a few frames
        self._set_callback(handler, frames_to_wait=4)


class Fill(_FillBase):
    """A container that automatically scales its content to fill the parent area.
    
    The Fill container adjusts its scaling_factor dynamically to ensure content
    fills as much of the parent as possible (up to 98% of width or height) without
    overshooting either dimension. Uses the minimum of width and height scale factors.
    
    Args:
        fill_percent (float): Target fill percentage (0.0 to 1.0). Default is 1.0 (100%).
            Use 0.8 for 80% fill, etc.
    
    Example:
        ```python
        with Fill():  # Content scales to full area (100%)
            dcg.Text(C, value="This will scale to fill the space")
            
        with Fill(0.8):  # Content scales to 80% of the area
            dcg.Text(C, value="This will scale to 80% of the space")
        ```
    
    Note:
        - The scaling converges over a few frames using a RenderHandler
        - Themes and fonts are reapplied when scaling changes for proper rendering
        - Target is 98% actual fill to allow for small variations in layout
    """
    
    @overload
    def __init__(self, context: dcg.Context, fill_percent: float = 1.0): ...
    
    @overload
    def __init__(self, fill_percent: float = 1.0): ...
    
    def __init__(self, *args, **kwargs):
        context, fill_percent = _parse_optional_arg_with_optional_context("fill_percent", 1.0, *args, **kwargs)
        super().__init__(context, fill_percent)
    
    def _calculate_scale(self, scale_x: float, scale_y: float) -> float:
        """Use the minimum scale to ensure we don't overshoot either dimension."""
        return min(scale_x, scale_y)


class FillV(_FillBase):
    """A container that automatically scales its content to fill the parent height.
    
    The FillV container adjusts its scaling_factor dynamically to ensure content
    fills the parent's height up to a specified percentage. Only considers vertical
    dimension for scaling.
    
    Args:
        fill_percent (float): Target fill percentage (0.0 to 1.0). Default is 1.0 (100%).
            Use 0.8 for 80% fill, etc.
    
    Example:
        ```python
        with FillV():  # Content scales to full height (100%)
            dcg.Text(C, value="This will scale to fill the height")
            
        with FillV(0.8):  # Content scales to 80% of the height
            dcg.Text(C, value="This will scale to 80% of the height")
        ```
    
    Note:
        - Only the vertical dimension is considered for scaling
        - Content may overflow horizontally if too wide
    """
    
    @overload
    def __init__(self, context: dcg.Context, fill_percent: float = 1.0): ...
    
    @overload
    def __init__(self, fill_percent: float = 1.0): ...
    
    def __init__(self, *args, **kwargs):
        context, fill_percent = _parse_optional_arg_with_optional_context("fill_percent", 1.0, *args, **kwargs)
        super().__init__(context, fill_percent)
    
    def _calculate_scale(self, scale_x: float, scale_y: float) -> float:
        """Use only the vertical scale factor."""
        return scale_y


class FillH(_FillBase):
    """A container that automatically scales its content to fill the parent width.
    
    The FillH container adjusts its scaling_factor dynamically to ensure content
    fills the parent's width up to a specified percentage. Only considers horizontal
    dimension for scaling.
    
    Args:
        fill_percent (float): Target fill percentage (0.0 to 1.0). Default is 1.0 (100%).
            Use 0.8 for 80% fill, etc.
    
    Example:
        ```python
        with FillH():  # Content scales to full width (100%)
            dcg.Text(C, value="This will scale to fill the width")
            
        with FillH(0.8):  # Content scales to 80% of the width
            dcg.Text(C, value="This will scale to 80% of the width")
        ```
    
    Note:
        - Only the horizontal dimension is considered for scaling
        - Content may overflow vertically if too tall
    """
    
    @overload
    def __init__(self, context: dcg.Context, fill_percent: float = 1.0): ...
    
    @overload
    def __init__(self, fill_percent: float = 1.0): ...
    
    def __init__(self, *args, **kwargs):
        context, fill_percent = _parse_optional_arg_with_optional_context("fill_percent", 1.0, *args, **kwargs)
        super().__init__(context, fill_percent)
    
    def _calculate_scale(self, scale_x: float, scale_y: float) -> float:
        """Use only the horizontal scale factor."""
        return scale_x


class Slide(dcg.ChildWindow):
    """A single slide in the presentation.
    
    This class represents a single slide within a slideshow presentation. It inherits from
    dcg.ChildWindow to provide a contained area for slide content.
    
    Attributes:
        border (bool): Whether to show a border around the slide (default: False)
        no_scrollbar (bool): Disables scrollbars (default: True)
        width (float): Width of the slide (0 means fill available width)
        height (float): Height of the slide (0 means fill available height)
        title (str): Title of the slide, displayed in the menubar
        margin (int): margin on each side of the slide (default: 30)
    
    Example:
        ```python
        with Slide(C, title="My First Slide"):
            dcg.Text(C, value="Hello World!")
        ```
    """
    @overload
    def __init__(self, context: dcg.Context, title: str = "", **kwargs): ...
    
    @overload
    def __init__(self, title: str = "", **kwargs): ...
    
    def __init__(self,
                 *args,
                 border: bool = False,
                 margin: float | None = 30.,
                 **kwargs):
        kwargs.setdefault("width", dcg.Size.FILLX())
        kwargs.setdefault("height", dcg.Size.FILLY())
        self.no_scrollbar = True
        context, title = _parse_positional_arg_with_optional_context("title", *args, **kwargs)
        self.label = title
        self._margin = margin
        self.no_scroll_with_mouse = True
        super().__init__(context, show=False, border=border, **kwargs)

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, _get_applicable_context(*args, **kwargs))

    def __exit__(self, a, b, c):
        super().__exit__(a, b, c)
        # Apply indentation to children
        margin = self._margin * dcg.Size.DPI()\
            if isinstance(self._margin, float)\
            else dcg.Size.THEME_STYLE("indent_spacing", False)
        children = self.children
        #for child in children:
        #    child.x = self.x.x1 + margin * dcg.Size.DPI() # TODO the other margin
        if len(children) > 0 and isinstance(children[-1], dcg.Layout) and children[-1].label == "footnote":
            children[-1].y = self.y.y2 - children[-1].height
        return False

class SlideSection(dcg.ChildWindow, metaclass=_ContextManagerMeta):
    """A sub-section container for organizing slide content.
    
    This class provides a clean container for grouping related content within a slide.
    It's particularly useful in conjunction with TwoColumnSlide for creating side-by-side
    content layouts.
    
    Attributes:
        no_scrollbar (bool): Disables scrollbars (default: True)
        border (bool): Whether to show a border (default: False)
    
    Example:
        ```python
        with TwoColumnSlide(C, title="Split Content"):
            with SlideSection(C):  # Left column
                dcg.Text(C, value="Left side content")
            with SlideSection(C):  # Right column
                dcg.Text(C, value="Right side content")
        ```
    """
    @overload
    def __init__(self, context: dcg.Context, **kwargs): ...
    
    @overload
    def __init__(self, **kwargs): ...
    
    def __init__(self, *args, **kwargs):
        # Handle both signatures
        context = _get_applicable_context(*args, **kwargs)
        
        self.no_scrollbar = True
        self.border = False
        self.no_scroll_with_mouse = True
        super().__init__(context, **kwargs)

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, _get_applicable_context(*args, **kwargs))


class Text(dcg.MarkDownText):
    """Markdown text"""
    @overload
    def __init__(self, context: dcg.Context, text: str, **kwargs): ...
    
    @overload
    def __init__(self, text: str, **kwargs): ...
    
    def __init__(self, *args, **kwargs):
        # Handle both signatures
        context, text = _parse_positional_arg_with_optional_context("text", *args, **kwargs)
        
        super().__init__(context, value=text, **kwargs)

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, _get_applicable_context(*args, **kwargs))


#################

class ThemeColorVariant(dcg.ThemeList):
    def __init__(self, C: dcg.Context, **kwargs):
        super().__init__(C, **kwargs)
        
        # Theme definitions with names
        with self:
            self.themes = [
                ("Dark", self._create_dark_theme()),
                ("Light", self._create_light_theme()),
                ("Sepia", self._create_sepia_theme()),
                ("Nord", self._create_nord_theme()),
                ("Dracula", self._create_dracula_theme()),
                ("Solarized Dark", self._create_solarized_dark_theme()),
                ("Ocean", self._create_ocean_theme()),
                ("Forest", self._create_forest_theme())
            ]
        
        self.current_index = 0
        self.current_theme = self.themes[0][1]
        self.children = [self.current_theme]

    def _create_dark_theme(self):
        return dcg.ThemeColorImGui(self.context,
            window_bg=(30, 30, 30),
            child_bg=(30, 30, 30),
            menu_bar_bg=(40, 40, 40),
            popup_bg=(35, 35, 35),
            border=(60, 60, 60),
            border_shadow=(0, 0, 0, 0),
            frame_bg=(50, 50, 50),
            frame_bg_hovered=(70, 70, 70),
            frame_bg_active=(90, 90, 90),
            title_bg=(40, 40, 40),
            title_bg_active=(50, 50, 50),
            title_bg_collapsed=(40, 40, 40),
            scrollbar_bg=(35, 35, 35),
            scrollbar_grab=(70, 70, 70),
            scrollbar_grab_hovered=(90, 90, 90),
            scrollbar_grab_active=(110, 110, 110),
            check_mark=(100, 100, 255),
            slider_grab=(100, 100, 255),
            slider_grab_active=(150, 150, 255),
            button=(60, 60, 60),
            button_hovered=(80, 80, 80),
            button_active=(100, 100, 100),
            header=(60, 60, 70),
            header_hovered=(70, 70, 80),
            header_active=(80, 80, 90),
            separator=(60, 60, 60),
            separator_hovered=(70, 70, 70),
            separator_active=(90, 90, 90),
            resize_grip=(60, 60, 60),
            resize_grip_hovered=(80, 80, 80),
            resize_grip_active=(100, 100, 100),
            tab=(50, 50, 50),
            tab_hovered=(70, 70, 70),
            tab_selected=(80, 80, 80),
            tab_dimmed=(40, 40, 40),
            tab_dimmed_selected=(50, 50, 50),
            plot_lines=(156, 156, 156),
            plot_lines_hovered=(255, 110, 89),
            plot_histogram=(230, 179, 0),
            plot_histogram_hovered=(255, 153, 0),
            text_selected_bg=(0, 90, 180, 180),
            drag_drop_target=(255, 255, 0, 230),
            nav_windowing_highlight=(255, 255, 255, 179),
            nav_windowing_dim_bg=(51, 51, 51, 179),
            modal_window_dim_bg=(35, 35, 35, 90),
            text=(230, 230, 230))

    def _create_light_theme(self):
        return dcg.ThemeColorImGui(self.context,
            window_bg=(245, 245, 245),
            child_bg=(245, 245, 245),
            menu_bar_bg=(235, 235, 235),
            popup_bg=(240, 240, 240),
            border=(200, 200, 200),
            border_shadow=(0, 0, 0, 0),
            frame_bg=(235, 235, 235),
            frame_bg_hovered=(225, 225, 225),
            frame_bg_active=(215, 215, 215),
            title_bg=(235, 235, 235),
            title_bg_active=(225, 225, 225),
            title_bg_collapsed=(235, 235, 235),
            scrollbar_bg=(240, 240, 240),
            scrollbar_grab=(190, 190, 190),
            scrollbar_grab_hovered=(170, 170, 170),
            scrollbar_grab_active=(150, 150, 150),
            check_mark=(100, 100, 255),
            slider_grab=(100, 100, 255),
            slider_grab_active=(80, 80, 255),
            button=(225, 225, 225),
            button_hovered=(215, 215, 215),
            button_active=(200, 200, 200),
            header=(220, 220, 230),
            header_hovered=(210, 210, 220),
            header_active=(200, 200, 210),
            separator=(200, 200, 200),
            separator_hovered=(190, 190, 190),
            separator_active=(180, 180, 180),
            resize_grip=(200, 200, 200),
            resize_grip_hovered=(190, 190, 190),
            resize_grip_active=(180, 180, 180),
            tab=(220, 220, 220),
            tab_hovered=(210, 210, 210),
            tab_selected=(200, 200, 200),
            tab_dimmed=(230, 230, 230),
            tab_dimmed_selected=(220, 220, 220),
            plot_lines=(100, 100, 100),
            plot_lines_hovered=(255, 110, 89),
            plot_histogram=(230, 179, 0),
            plot_histogram_hovered=(255, 153, 0),
            text_selected_bg=(173, 214, 255),
            drag_drop_target=(255, 255, 0, 230),
            nav_windowing_highlight=(0, 0, 0, 179),
            nav_windowing_dim_bg=(204, 204, 204, 179),
            modal_window_dim_bg=(204, 204, 204, 90),
            text=(30, 30, 30))

    def _create_sepia_theme(self):
        return dcg.ThemeColorImGui(self.context,
            window_bg=(251, 240, 217),
            child_bg=(251, 240, 217),
            menu_bar_bg=(242, 229, 201),
            popup_bg=(251, 240, 217),
            border=(200, 186, 157),
            border_shadow=(0, 0, 0, 0),
            frame_bg=(242, 229, 201),
            frame_bg_hovered=(236, 221, 188),
            frame_bg_active=(229, 212, 175),
            title_bg=(242, 229, 201),
            title_bg_active=(236, 221, 188),
            title_bg_collapsed=(242, 229, 201),
            scrollbar_bg=(242, 229, 201),
            scrollbar_grab=(220, 201, 159),
            scrollbar_grab_hovered=(200, 186, 157),
            scrollbar_grab_active=(180, 166, 137),
            check_mark=(180, 166, 137),
            slider_grab=(200, 186, 157),
            slider_grab_active=(180, 166, 137),
            button=(236, 221, 188),
            button_hovered=(229, 212, 175),
            button_active=(220, 201, 159),
            header=(236, 221, 188),
            header_hovered=(229, 212, 175),
            header_active=(220, 201, 159),
            separator=(200, 186, 157),
            separator_hovered=(180, 166, 137),
            separator_active=(160, 146, 117),
            resize_grip=(220, 201, 159),
            resize_grip_hovered=(200, 186, 157),
            resize_grip_active=(180, 166, 137),
            tab=(236, 221, 188),
            tab_hovered=(229, 212, 175),
            tab_selected=(220, 201, 159),
            tab_dimmed=(242, 229, 201),
            tab_dimmed_selected=(236, 221, 188),
            plot_lines=(160, 146, 117),
            plot_lines_hovered=(180, 166, 137),
            plot_histogram=(200, 186, 157),
            plot_histogram_hovered=(180, 166, 137),
            text_selected_bg=(200, 186, 157, 180),
            drag_drop_target=(180, 166, 137, 230),
            nav_windowing_highlight=(200, 186, 157, 179),
            nav_windowing_dim_bg=(251, 240, 217, 179),
            modal_window_dim_bg=(251, 240, 217, 90),
            text=(101, 80, 40))

    def _create_nord_theme(self):
        return dcg.ThemeColorImGui(self.context,
            window_bg=(46, 52, 64),
            child_bg=(46, 52, 64),
            menu_bar_bg=(59, 66, 82),
            popup_bg=(46, 52, 64),
            border=(76, 86, 106),
            border_shadow=(0, 0, 0, 0),
            frame_bg=(67, 76, 94),
            frame_bg_hovered=(76, 86, 106),
            frame_bg_active=(86, 97, 119),
            title_bg=(59, 66, 82),
            title_bg_active=(67, 76, 94),
            title_bg_collapsed=(59, 66, 82),
            scrollbar_bg=(46, 52, 64),
            scrollbar_grab=(76, 86, 106),
            scrollbar_grab_hovered=(86, 97, 119),
            scrollbar_grab_active=(93, 104, 126),
            check_mark=(136, 192, 208),
            slider_grab=(129, 161, 193),
            slider_grab_active=(136, 192, 208),
            button=(67, 76, 94),
            button_hovered=(76, 86, 106),
            button_active=(86, 97, 119),
            header=(67, 76, 94),
            header_hovered=(76, 86, 106),
            header_active=(86, 97, 119),
            separator=(76, 86, 106),
            separator_hovered=(86, 97, 119),
            separator_active=(93, 104, 126),
            resize_grip=(76, 86, 106),
            resize_grip_hovered=(86, 97, 119),
            resize_grip_active=(93, 104, 126),
            tab=(67, 76, 94),
            tab_hovered=(76, 86, 106),
            tab_selected=(86, 97, 119),
            tab_dimmed=(59, 66, 82),
            tab_dimmed_selected=(67, 76, 94),
            plot_lines=(136, 192, 208),
            plot_lines_hovered=(143, 188, 187),
            plot_histogram=(129, 161, 193),
            plot_histogram_hovered=(136, 192, 208),
            text_selected_bg=(76, 86, 106, 180),
            drag_drop_target=(136, 192, 208, 230),
            nav_windowing_highlight=(129, 161, 193, 179),
            nav_windowing_dim_bg=(46, 52, 64, 179),
            modal_window_dim_bg=(46, 52, 64, 90),
            text=(236, 239, 244))

    def _create_dracula_theme(self):
        return dcg.ThemeColorImGui(self.context,
            window_bg=(40, 42, 54),
            child_bg=(30, 30, 30),
            menu_bar_bg=(68, 71, 90),
            popup_bg=(40, 42, 54),
            border=(98, 114, 164),
            border_shadow=(0, 0, 0, 0),
            frame_bg=(68, 71, 90),
            frame_bg_hovered=(78, 82, 104),
            frame_bg_active=(88, 91, 112),
            title_bg=(68, 71, 90),
            title_bg_active=(78, 82, 104),
            title_bg_collapsed=(68, 71, 90),
            scrollbar_bg=(40, 42, 54),
            scrollbar_grab=(68, 71, 90),
            scrollbar_grab_hovered=(78, 82, 104),
            scrollbar_grab_active=(88, 91, 112),
            check_mark=(189, 147, 249),
            slider_grab=(189, 147, 249),
            slider_grab_active=(255, 121, 198),
            button=(68, 71, 90),
            button_hovered=(78, 82, 104),
            button_active=(88, 91, 112),
            header=(68, 71, 90),
            header_hovered=(78, 82, 104),
            header_active=(88, 91, 112),
            separator=(98, 114, 164),
            separator_hovered=(108, 124, 174),
            separator_active=(118, 134, 184),
            resize_grip=(68, 71, 90),
            resize_grip_hovered=(78, 82, 104),
            resize_grip_active=(88, 91, 112),
            tab=(68, 71, 90),
            tab_hovered=(78, 82, 104),
            tab_selected=(88, 91, 112),
            tab_dimmed=(40, 42, 54),
            tab_dimmed_selected=(68, 71, 90),
            plot_lines=(248, 248, 242),
            plot_lines_hovered=(255, 121, 198),
            plot_histogram=(189, 147, 249),
            plot_histogram_hovered=(255, 121, 198),
            text_selected_bg=(98, 114, 164, 180),
            drag_drop_target=(189, 147, 249, 230),
            nav_windowing_highlight=(189, 147, 249, 179),
            nav_windowing_dim_bg=(40, 42, 54, 179),
            modal_window_dim_bg=(40, 42, 54, 90),
            text=(248, 248, 242))

    def _create_solarized_dark_theme(self):
        return dcg.ThemeColorImGui(self.context,
            window_bg=(0, 43, 54),
            child_bg=(0, 43, 54),
            menu_bar_bg=(7, 54, 66),
            popup_bg=(0, 43, 54),
            border=(88, 110, 117),
            border_shadow=(0, 0, 0, 0),
            frame_bg=(7, 54, 66),
            frame_bg_hovered=(23, 66, 77),
            frame_bg_active=(32, 77, 87),
            title_bg=(7, 54, 66),
            title_bg_active=(23, 66, 77),
            title_bg_collapsed=(7, 54, 66),
            scrollbar_bg=(0, 43, 54),
            scrollbar_grab=(88, 110, 117),
            scrollbar_grab_hovered=(101, 123, 131),
            scrollbar_grab_active=(131, 148, 150),
            check_mark=(181, 137, 0),
            slider_grab=(133, 153, 0),
            slider_grab_active=(181, 137, 0),
            button=(7, 54, 66),
            button_hovered=(23, 66, 77),
            button_active=(32, 77, 87),
            header=(7, 54, 66),
            header_hovered=(23, 66, 77),
            header_active=(32, 77, 87),
            separator=(88, 110, 117),
            separator_hovered=(101, 123, 131),
            separator_active=(131, 148, 150),
            resize_grip=(88, 110, 117),
            resize_grip_hovered=(101, 123, 131),
            resize_grip_active=(131, 148, 150),
            tab=(7, 54, 66),
            tab_hovered=(23, 66, 77),
            tab_selected=(32, 77, 87),
            tab_dimmed=(0, 43, 54),
            tab_dimmed_selected=(7, 54, 66),
            plot_lines=(147, 161, 161),
            plot_lines_hovered=(181, 137, 0),
            plot_histogram=(133, 153, 0),
            plot_histogram_hovered=(181, 137, 0),
            text_selected_bg=(88, 110, 117, 180),
            drag_drop_target=(133, 153, 0, 230),
            nav_windowing_highlight=(133, 153, 0, 179),
            nav_windowing_dim_bg=(0, 43, 54, 179),
            modal_window_dim_bg=(0, 43, 54, 90),
            text=(147, 161, 161))

    def _create_ocean_theme(self):
        return dcg.ThemeColorImGui(self.context,
            window_bg=(28, 45, 65),
            child_bg=(28, 45, 65),
            menu_bar_bg=(36, 55, 77),
            popup_bg=(28, 45, 65),
            border=(52, 74, 100),
            border_shadow=(0, 0, 0, 0),
            frame_bg=(36, 55, 77),
            frame_bg_hovered=(52, 74, 100),
            frame_bg_active=(64, 89, 119),
            title_bg=(36, 55, 77),
            title_bg_active=(52, 74, 100),
            title_bg_collapsed=(36, 55, 77),
            scrollbar_bg=(28, 45, 65),
            scrollbar_grab=(52, 74, 100),
            scrollbar_grab_hovered=(64, 89, 119),
            scrollbar_grab_active=(77, 106, 141),
            check_mark=(103, 183, 255),
            slider_grab=(71, 161, 241),
            slider_grab_active=(103, 183, 255),
            button=(36, 55, 77),
            button_hovered=(52, 74, 100),
            button_active=(64, 89, 119),
            header=(36, 55, 77),
            header_hovered=(52, 74, 100),
            header_active=(64, 89, 119),
            separator=(52, 74, 100),
            separator_hovered=(64, 89, 119),
            separator_active=(77, 106, 141),
            resize_grip=(52, 74, 100),
            resize_grip_hovered=(64, 89, 119),
            resize_grip_active=(77, 106, 141),
            tab=(36, 55, 77),
            tab_hovered=(52, 74, 100),
            tab_selected=(64, 89, 119),
            tab_dimmed=(28, 45, 65),
            tab_dimmed_selected=(36, 55, 77),
            plot_lines=(154, 206, 255),
            plot_lines_hovered=(103, 183, 255),
            plot_histogram=(71, 161, 241),
            plot_histogram_hovered=(103, 183, 255),
            text_selected_bg=(52, 74, 100, 180),
            drag_drop_target=(71, 161, 241, 230),
            nav_windowing_highlight=(71, 161, 241, 179),
            nav_windowing_dim_bg=(28, 45, 65, 179),
            modal_window_dim_bg=(28, 45, 65, 90),
            text=(192, 215, 235))

    def _create_forest_theme(self):
        return dcg.ThemeColorImGui(self.context,
            window_bg=(35, 45, 35),
            child_bg=(30, 30, 30),
            menu_bar_bg=(45, 55, 45),
            popup_bg=(35, 45, 35),
            border=(65, 85, 65),
            border_shadow=(0, 0, 0, 0),
            frame_bg=(45, 55, 45),
            frame_bg_hovered=(55, 75, 55),
            frame_bg_active=(65, 85, 65),
            title_bg=(45, 55, 45),
            title_bg_active=(55, 75, 55),
            title_bg_collapsed=(45, 55, 45),
            scrollbar_bg=(35, 45, 35),
            scrollbar_grab=(65, 85, 65),
            scrollbar_grab_hovered=(75, 95, 75),
            scrollbar_grab_active=(85, 105, 85),
            check_mark=(141, 197, 62),
            slider_grab=(126, 179, 51),
            slider_grab_active=(141, 197, 62),
            button=(45, 55, 45),
            button_hovered=(55, 75, 55),
            button_active=(65, 85, 65),
            header=(45, 55, 45),
            header_hovered=(55, 75, 55),
            header_active=(65, 85, 65),
            separator=(65, 85, 65),
            separator_hovered=(75, 95, 75),
            separator_active=(85, 105, 85),
            resize_grip=(65, 85, 65),
            resize_grip_hovered=(75, 95, 75),
            resize_grip_active=(85, 105, 85),
            tab=(45, 55, 45),
            tab_hovered=(55, 75, 55),
            tab_selected=(65, 85, 65),
            tab_dimmed=(35, 45, 35),
            tab_dimmed_selected=(45, 55, 45),
            plot_lines=(180, 210, 120),
            plot_lines_hovered=(141, 197, 62),
            plot_histogram=(126, 179, 51),
            plot_histogram_hovered=(141, 197, 62),
            text_selected_bg=(65, 85, 65, 180),
            drag_drop_target=(126, 179, 51, 230),
            nav_windowing_highlight=(126, 179, 51, 179),
            nav_windowing_dim_bg=(35, 45, 35, 179),
            modal_window_dim_bg=(35, 45, 35, 90),
            text=(210, 230, 190))

    def toggle(self):
        """Cycle to the next available theme"""
        self.current_index = (self.current_index + 1) % len(self.themes)
        self.current_theme = self.themes[self.current_index][1]
        self.children = [self.current_theme]
        return self.themes[self.current_index][0]  # Return theme name

def _get_current_theme(item: dcg.uiItem):
    """
    Get the current theme for an item.
    
    Walks up the parent chain to find the first theme assigned.
    """
    current = item
    while current is not None:
        if current.theme is not None:
            return current.theme
        current = current.parent
    return None

def _get_current_font(item: dcg.uiItem) -> dcg.Font | None:
    """
    Get the current font for an item.
    
    Walks up the parent chain to find the first font assigned,
    up to the viewport level where the default font is retrieved.
    """
    current = item
    while current is not None:
        if current.font is not None:
            return current.font
        current = current.parent
    return None

class SlideShow(dcg.Window):
    """
    This class manages the entire presentation, including slide navigation,
    theming, and display controls.
    
    Args:
        title (str): Title of the presentation window
        **kwargs: Additional arguments passed to dcg.Window
    
    Features:
        - Slide navigation (arrow keys, buttons)
        - Multiple color themes
        - Fullscreen toggle
        - Progress bar
        - Presentation scaling
        - PDF export capability
    
    Example:
        ```python
        # Create a basic presentation
        C = dcg.Context()
        slideshow = SlideShow(C, title="My Presentation")
        
        with slideshow:
            with Slide(C, title="Introduction"):
                dcg.Text(C, value="Welcome to my presentation!")
                
            with Slide(C, title="Content"):
                dcg.Text(C, value="Some content here...")
        
        # Initialize and run
        C.viewport.initialize()
        slideshow.start()
        while C.running:
            C.viewport.render_frame()
        ```
    
    Controls:
        - Left/Right arrow keys: Navigate slides
        - F: Toggle fullscreen
        - T: Toggle color theme
        - \\: Adjust presentation scale
        - Esc: Exit presentation
        - Right-click menubar: Export to PDF
    """
    
    def __init__(self, C: dcg.Context, title: str = "Slideshow", **kwargs):
        super().__init__(C, label=title, no_collapse=True, **kwargs)
        self.slides: list['Slide'] = []
        self.current_slide: int = 0
        self.primary = True
        self.no_move = True
        self.no_title_bar = True
        self.padding = (10, 10)
        self._slide_action_counter = 0  # Track slide changes for timer callbacks
        
        # Setup dark theme
        with dcg.ThemeList(self.context) as theme:
            self.theme_color = ThemeColorVariant(self.context)
            style_imgui = \
                dcg.ThemeStyleImGui(self.context,
                    window_padding=(10, 10),
                    frame_padding=(6, 3),
                    item_spacing=(8, 6),
                    scrollbar_size=12,
                    grab_min_size=20,
                    window_border_size=1,
                    child_border_size=1,
                    frame_border_size=0,
                    window_rounding=0,
                    frame_rounding=4,
                    popup_rounding=4,
                    scrollbar_rounding=4,
                    grab_rounding=4)
            style_implot = dcg.ThemeStyleImPlot(self.context)

        # Since we change the scaling factor, in order to have it apply
        # to all theme values, we need to reapply the theme entirely.
        # thus we fill in the theme with the default values
        for name in dir(dcg.ThemeStyleImGui):
            try:
                if getattr(style_imgui, name, None) is None:
                    setattr(style_imgui, name, style_imgui.get_default(name))
            except:
                pass
        for name in dir(dcg.ThemeStyleImPlot):
            try:
                setattr(style_implot, name, style_implot.get_default(name))
            except:
                pass

        self.theme = theme
        self.font = _get_current_font(self)
        if self.font is None:
            self.font = dcg.AutoFont.get_default(self.context)

        self._setup_menubar()
        self._setup_progress_bar()
        self._setup_slide_reference()

    def _setup_menubar(self):
        """Setup the top menubar with navigation controls"""
        # Add right-click handler to menubar
        def show_save_popup():
            with dcg.Window(self.context, popup=True) as popup:
                def quick_save():
                    popup.show = False
                    def export_thread():
                        export_slideshow(_get_unique_filename("slides", "pdf"), self)
                    thread = threading.Thread(target=export_thread)
                    thread.start()
            
                def save_as():
                    popup.show = False
                    def save_callback(paths):
                        if paths:
                            def export_thread():
                                export_slideshow(paths[0], self)
                            thread = threading.Thread(target=export_thread)
                            thread.start()
                    dcg.os.show_save_file_dialog(self.context, save_callback)

                dcg.Button(self.context, label="Quick save", callback=quick_save)
                dcg.Button(self.context, label="Save as...", callback=save_as)

        with dcg.MenuBar(self.context, parent=self):
            # Navigation buttons with icons
            dcg.Button(self.context, arrow=dcg.ButtonDirection.LEFT,
                       callback=self.previous_slide)
            dcg.Button(self.context, arrow=dcg.ButtonDirection.RIGHT,
                       callback=self.next_slide)
                
            dcg.Spacer(self.context, label="|")
                
            # Scale button that opens popup
            scale_button = dcg.Button(self.context, label="\\", small=True, width=40)
                
            def open_scale_popup(sender):
                slideshow = sender.parent.parent
                with dcg.Window(self.context, popup=True,
                                no_move=True,
                                font=slideshow.font,
                                scaling_factor=slideshow.scaling_factor,
                                width=200, autosize=True):
                    dcg.Text(self.context, value="Presentation Scale")
                    dcg.Separator(self.context)
                    dcg.Slider(self.context,
                               value=slideshow.scaling_factor,
                               min_value=0.5,
                               max_value=3.0,
                               keyboard_clamped=True,
                               width=150,
                               callback=self._update_scale)
                
            scale_button.callback = open_scale_popup
                
            # Fullscreen and theme toggles
            dcg.Button(self.context, label="F", callback=self._toggle_fullscreen,
                       small=True, width=40)
            dcg.Button(self.context, label="T", callback=self._toggle_theme,
                       small=True, width=40)
            with dcg.Tooltip(self.context):
                self._theme_label = dcg.Text(self.context, value="Dark")
                
            dcg.Spacer(self.context, label="|")
                
            # Slide counter and title with better styling
            self._slide_counter = dcg.Text(self.context, value="0/0")
            dcg.Spacer(self.context, width=20)
            self._slide_title = dcg.Text(self.context, value="")
        def end_presentation():
            self.context.running = False
        key_handlers = dcg.HandlerList(self.context)
        with key_handlers:
            dcg.KeyPressHandler(self.context, callback=self.previous_slide, key=dcg.Key.LEFTARROW)
            dcg.KeyPressHandler(self.context, callback=self.next_slide, key=dcg.Key.RIGHTARROW)
            dcg.MouseClickHandler(self.context, 
                callback=show_save_popup,
                button=dcg.MouseButton.RIGHT)

        # Do not steal key events of children items
        # That also means the menubar has to be clicked (to regain focus)
        # to have keys having effect
        key_if_focused = dcg.ConditionalHandler(self.context)
        key_handlers.parent = key_if_focused
        dcg.FocusHandler(self.context, parent=key_if_focused)
    
        self.handlers += [
            key_if_focused,
            dcg.KeyPressHandler(self.context, callback=end_presentation, key=dcg.Key.ESCAPE)
            ]

    def _setup_progress_bar(self):
        """Create progress bar at bottom of window"""
        with dcg.HorizontalLayout(self.context, parent=self):
            #dcg.Spacer(self.context, height=5)
            self._progress_bar = dcg.ProgressBar(self.context, 
                                                 value=0.0,
                                                 overlay="",
                                                 width=dcg.Size.FILLX(),
                                                 height=3)
            #dcg.Spacer(self.context, height=5)

    def _setup_slide_reference(self):
        """Create invisible background overlay for slide transitions"""
        # Create invisible background overlay for slide transitions
        # This stays on the background is used as reference for slide positions
        self._slides_invisible_background = dcg.ChildWindow(
            self.context,
            parent=self,
            width=dcg.Size.FILLX(),
            height=dcg.Size.FILLY(),
            border=False,
            no_scrollbar=True,
            no_scroll_with_mouse=True
        )

    def __enter__(self):
        """Push context onto stack when entering with statement"""
        _push_context(self.context)
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Pop context from stack when exiting with statement"""
        result = super().__exit__(exc_type, exc_val, exc_tb)
        _pop_context()
        return result

    def _toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        self.context.viewport.fullscreen = not self.context.viewport.fullscreen

    def _toggle_theme(self):
        """Toggle between color themes"""
        theme_name = self.theme_color.toggle()
        self._theme_label.value = theme_name

    def _update_slide_counter(self):
        """Update the slide counter text"""
        self._slide_counter.value = f"Slide: {self.current_slide + 1}/{len(self.slides)}"

    def _update_scale(self, sender, target, value: float):
        """Update the font scale"""
        self.scaling_factor = value

    def show_slide(self, index: int):
        """Show the slide at the given index with proper convergence handling"""
        if 0 <= index < len(self.slides):
            self.current_slide = index
            
            # Increment action counter to invalidate old timer callbacks
            self._slide_action_counter += 1
            current_action = self._slide_action_counter

            # Show current slide and put it on top of others
            self.slides[index].show = True
            self.slides[index].parent = self.slides[index].parent

            # Hide all other slides
            for i, slide in enumerate(self.slides):
                if i != index:
                    slide.show = False

            # Show neighboring slides (previous and next) for size/font convergence
            # They will be hidden after 2.0s
            if index > 0:
                self.slides[index - 1].show = True # Note: is below current slide
            if index < len(self.slides) - 1:
                self.slides[index + 1].show = True # Same here
            
            # Update UI elements
            self._update_slide_counter()
            self._slide_title.value = dcg.make_bold_italic(self.slides[index].label)
            self._progress_bar.value = (index + 1) / len(self.slides)
            self._progress_bar.overlay = f"{index + 1}/{len(self.slides)}"

            # Start timer to hide neighbors after convergence, to prevent button interaction issues
            def convergence_timer(current_action=current_action, index=index):
                time.sleep(2.)
                # Only execute if slide hasn't changed (action counter matches)
                if self._slide_action_counter == current_action:
                    # Hide neighboring slides to avoid interaction issues
                    for i, slide in enumerate(self.slides):
                        if i != index:
                            slide.show = False
                    self.context.viewport.wake()
            
            thread = threading.Thread(target=convergence_timer, daemon=True)
            thread.start()
            
            self.context.viewport.wake()

    def next_slide(self):
        """Move to the next slide"""
        if self.current_slide < len(self.slides) - 1:
            self.show_slide(self.current_slide + 1)

    def previous_slide(self):
        """Move to the previous slide"""
        if self.current_slide > 0:
            self.show_slide(self.current_slide - 1)

    def start(self):
        """Show the first slide"""
        # Retrieve the slides from the children list
        self.slides = [child for child in self.children if isinstance(child, Slide)]
        
        # Configure all slides to match the invisible background positioning
        # This ensures proper overlapping during transitions
        bg = self._slides_invisible_background
        for slide in self.slides:
            slide.x = bg.x
            slide.y = bg.y
            slide.width = bg.width
            slide.height = bg.height
        
        # Show invisible background on top initially to hide first slide convergence
        self._slides_invisible_background.parent = self
        
        self.show_slide(0)
        
        # Hide the invisible background after 0.2s to reveal the converged slide
        def hide_background_timer():
            time.sleep(0.2)
            # put behind the other slides
            self._slides_invisible_background.previous_sibling = self._progress_bar.parent

        thread = threading.Thread(target=hide_background_timer, daemon=True)
        thread.start()