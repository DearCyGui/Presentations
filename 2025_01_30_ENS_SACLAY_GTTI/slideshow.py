import dearcygui as dcg
from typing import Optional, List, Tuple
import time
import io
import sys
from contextlib import redirect_stdout
import numpy as np
import os.path

from dearcygui.utils.image import DrawTiledImage, DrawSVG
import math
import requests
import threading
from PIL import Image

import dearcygui as dcg
from dearcygui.font import make_bold, make_bold_italic, make_italic

from pygments import highlight
from pygments.formatters import Terminal256Formatter
from pygments.lexers import get_lexer_by_name, guess_lexer
from pygments.util import ClassNotFound
from stransi import Ansi, SetAttribute, SetColor
from stransi.attribute import Attribute as AnsiAttribute
from stransi.color import ColorRole as AnsiColorRole

import marko
import os
import time
import imageio

##### Made with DearCyGui 0.0.10


##### Set of utilities that will probably one day
##### end up in dearcygui.utils

class DrawGif(dcg.utils.DrawStream):
    def __init__(self, context, gif_path, pmin=(0., 0.), pmax=(1., 1.), **kwargs):
        super().__init__(context, **kwargs)
        total_duration = 0
        
        # Load the GIF and extract frames
        gif = Image.open(gif_path)
        try:
            while True:
                # Convert frame to RGBA
                frame = np.array(gif.convert('RGBA'))
                # Create texture from frame
                texture = dcg.Texture(context)
                texture.set_value(frame)
                
                # Get frame duration in seconds
                frame_duration = gif.info['duration'] / 1000.0
                total_duration += frame_duration
                
                # Create DrawImage for this frame and add to stream
                image = dcg.DrawImage(context,
                                    texture=texture,
                                    pmin=pmin,
                                    pmax=pmax,
                                    parent=self)
                self.push(image, total_duration)
                
                # Move to next frame
                gif.seek(gif.tell() + 1)
        except EOFError:
            pass

        # Set stream to loop
        self.time_modulus = total_duration

class Gif(dcg.DrawInWindow):
    def __init__(self, context, gif_path, width=16, height=16, **kwargs):
        super().__init__(context, button=True, width=width, height=height, **kwargs)
        self.relative = True
        DrawGif(context, gif_path, parent=self)


def blinking_callback(sender, item):
    t = int(time.time())
    text_color = item.theme.children[0].Text
    c = dcg.color_as_floats(text_color)
    # Alternate between transparent and full
    if t % 2 == 0:
        c = (c[0], c[1], c[2], 0)
    else:
        c = (c[0], c[1], c[2], 1.)
    item.theme.children[0].Text = c
    item.context.viewport.wake()

class TextAnsi(dcg.HorizontalLayout):
    """
    Similar to dcg.Text, but has a limited support
    for ANSI escape sequences.
    Unlike dcg.Text, newlines are not supported.
    """
    def __init__(self, context, wrap=0, **kwargs):
        self.textline = ""
        self._bullet = False
        self.theme = dcg.ThemeStyleImGui(self.context, ItemSpacing=(0, 0))
        super().__init__(context, width=wrap, **kwargs)

    def render_text(self):
        self.children = [] # detach any previous text
        color = (255, 255, 255, 255) # Start with white
        bold = False
        italic = False
        background_color = None
        blinking = False
        underline = False # TODO
        strikethrough = False # TODO
        with self:
            if self._bullet:
                dcg.Text(self.context, bullet=True, value="")
            for instr in Ansi(self.textline).instructions():
                if isinstance(instr, SetAttribute):
                    if instr.attribute == AnsiAttribute.NORMAL:
                        bold = False
                        italic = False
                        background_color = None
                        blinking = False
                        underline = False
                        strikethrough = False
                    elif instr.attribute == AnsiAttribute.BOLD:
                        bold = True
                    elif instr.attribute == AnsiAttribute.ITALIC:
                        italic = True
                    elif instr.attribute == AnsiAttribute.UNDERLINE:
                        underline = True
                    elif instr.attribute == AnsiAttribute.BLINK:
                        blinking = True
                    elif instr.attribute == AnsiAttribute.NEITHER_BOLD_NOR_DIM:
                        bold = False
                    elif instr.attribute == AnsiAttribute.NOT_ITALIC:
                        italic = False
                    elif instr.attribute == AnsiAttribute.NOT_UNDERLINE:
                        underline = False
                    elif instr.attribute == AnsiAttribute.NOT_BLINK:
                        blinking = False
                    else:
                        raise RuntimeWarning("Unparsed Ansi: ", instr)
                elif isinstance(instr, SetColor):
                    if instr.role == AnsiColorRole.BACKGROUND:
                        if instr.color is None:
                            background_color = None
                        else:
                            background_color = instr.color.rgb
                            background_color = (background_color.red, background_color.green, background_color.blue)
                        continue
                    if instr.color is None:
                        # reset color
                        color = (255, 255, 255, 255)
                        continue
                    color = instr.color.rgb
                    color = (color.red, color.green, color.blue)
                elif isinstance(instr, str):
                    text = instr
                    if bold and italic:
                        text = make_bold_italic(text)
                    elif italic:
                        text = make_italic(text)
                    elif bold:
                        text = make_bold(text)
                    words = text.split(" ")
                    if background_color is None and not(blinking):
                        # add a space at the end of each words,
                        # except the last one.
                        words = [w + " " for w in words[:-1]] + words[-1:]
                        for word in words:
                            dcg.Text(self.context, value=word, color=color)
                    else:
                        current_theme = dcg.ThemeList(self.context)
                        current_theme_style = dcg.ThemeStyleImGui(self.context,
                                                  ItemSpacing=(0, 0),
                                                  FrameBorderSize=0,
                                                  FramePadding=(0, 0),
                                                  FrameRounding=0,
                                                  ItemInnerSpacing=(0, 0))
                        current_theme_color = dcg.ThemeColorImGui(self.context)
                        current_theme.children = [current_theme_color, current_theme_style]
                        bg_color = background_color if background_color is not None else (0, 0, 0, 0)
                        current_theme_color.Button = bg_color
                        current_theme_color.ButtonHovered = bg_color
                        current_theme_color.ButtonActive = bg_color
                        current_theme_color.Text = color
                        words = [w + " " for w in words[:-1]] + words[-1:]
                        # Wrapping the text within a button window.
                        for word in words:
                            dcg.Button(self.context,
                                       label=word,
                                       small=True,
                                       theme=current_theme,
                                       handlers=dcg.RenderHandler(self.context, callback=blinking_callback) if blinking else [])

                else:
                    raise RuntimeWarning("Unparsed Ansi: ", instr)

    @property
    def bullet(self):
        return self._bullet

    @bullet.setter
    def bullet(self, value):
        self._bullet = value
        self.render_text()

    @property
    def value(self):
        return self.textline

    @value.setter
    def value(self, value):
        self.textline = value
        self.render_text()


color_to_ansi = {
    "black": "90",
    "red": "91",
    "green": "92",
    "yellow": "93",
    "blue": "94",
    "magenta": "95",
    "cyan": "96",
    "white": "97"
}

def make_color(text : str, color : str | list = "white"):
    """
    Add ANSI escape codes to a text to render in color
    using TextAnsi.
    text: the text string to color
    color: the color name or color code
        Supported names are black, red, green, yellow, blue,
        magenta, cyan and white
        Else a color in any dcg color format is supported.
    """
    escape = "\u001b"
    if isinstance(color, str):
        transformed = f"{escape}[{color_to_ansi[color]}m{text}{escape}[39m"
    else:
        color = dcg.color_as_ints(color)
        (r, g, b, _) = color
        transformed = f"{escape}[38;2;{r};{g};{b}m{text}{escape}[39m"
    return transformed

def make_bg_color(text : str, color : str | list = "white"):
    """
    Add ANSI escape codes to add a colored background to text
    using TextAnsi.
    text: the text string to color
    color: the color name or color code
        Supported names are black, red, green, yellow, blue,
        magenta, cyan and white
        Else a color in any dcg color format is supported.
    """
    escape = "\u001b"
    if isinstance(color, str):
        transformed = f"{escape}[{str(int(color_to_ansi[color])+10)}m{text}{escape}[49m"
    else:
        color = dcg.color_as_ints(color)
        (r, g, b, _) = color
        transformed = f"{escape}[48;2;{r};{g};{b}m{text}{escape}[49m"
    return transformed

def make_blinking(text : str):
    """
    Add ANSI escape codes to make a text blinking
    using TextAnsi.
    text: the text string to blink
    """
    escape = "\u001b"
    transformed = f"{escape}[5m{text}{escape}[25m"
    return transformed

class MarkDownText(dcg.Layout, marko.Renderer):
    """
    Text displayed in DearCyGui using Marko to render

    Will use the viewport font or the font passed in the 
    initialization arguments.
    """
    def __init__(self, C : dcg.Context, wrap : int = 0, **kwargs):
        """
        C: the context
        wrap: Text() wrap attribute. 0 means
            wrap at the end of the window. > 0 means
            a specified size.
        """
        self.C = C

        self.font = kwargs.pop("font", self.context.viewport.font)
        if isinstance(self.font, dcg.AutoFont):
            # We will cheat by using the AutoFont feature
            # to build various scales for us.
            # This enables to avoid duplicating fonts if we
            # have several MarkDownText instances.
            self.huge_font_scale = 2.
            self.big_font_scale = 1.5
            self.use_auto_scale = True
        else:
            self.huge_font = dcg.AutoFont(C, 34)
            self.big_font = dcg.AutoFont(C, 25)
            self.use_auto_scale = False
        self.default_font = C.viewport.font
        self.wrap = wrap
        self.no_spacing = dcg.ThemeStyleImGui(C, FramePadding=(0,0), FrameBorderSize=0, ItemSpacing=(0, 0))
        self._text = None
        marko.Renderer.__init__(self)
        dcg.Layout.__init__(self, C, **kwargs)

    @property
    def value(self):
        """Content text"""
        return self._text

    @value.setter
    def value(self, text):
        if not(isinstance(text, str)):
            raise ValueError("Expected a string as text")
        self._text = text
        parsed_text = marko.Markdown().parse(text)
        with self:
            self.render(parsed_text)

    def render_children_if_not_str(self, element):
        if isinstance(element, str):
            return element
        elif isinstance(element.children, str):
            return element.children
        else:
            return self.render_children(element)

    def render_document(self, element):
        text = self.render_children_if_not_str(element)
        if text != "":
            TextAnsi(self.C, wrap=self.wrap, value=text)
        return ""

    def render_paragraph(self, element):
        with dcg.VerticalLayout(self.C):
            text = self.render_children_if_not_str(element)
            if text != "":
                TextAnsi(self.C, wrap=self.wrap, value=text)
        dcg.Spacer(self.C)
        return ""

    def render_list(self, element):
        with dcg.VerticalLayout(self.C, indent=-1):
            self.render_children_if_not_str(element)
        return ""

    def render_list_item(self, element):
        with dcg.Layout(self.C, theme=self.no_spacing) as l:
            with dcg.VerticalLayout(self.C) as vl:
                text = self.render_children_if_not_str(element)
                if text != "":
                    TextAnsi(self.C, bullet=True, value="text")
                else:
                    # text rendered inside render_children_if_not_str
                    # insert the bullet
                    l.children = [TextAnsi(self.C, wrap=self.wrap, bullet=True, no_newline=True, value="", attach=False), vl]
        dcg.Spacer(self.C) # TODO: somehow the no_spacing theme affects the spacer !
        dcg.Spacer(self.C)
        dcg.Spacer(self.C)
        return ""

    def render_quote(self, element):
        with dcg.ChildWindow(self.C, width=0, height=0):
            text = self.render_children_if_not_str(element)
            if text != "":
                TextAnsi(self.C, bullet=True, value=make_italic(text))
        return ""

    def render_fenced_code(self, element):
        code = element.children[0].children
        if element.lang:
            try:
                lexer = get_lexer_by_name(element.lang, stripall=True, encoding='utf-8')
            except ClassNotFound:
                lexer = guess_lexer(code, encoding='utf-8')
        else:
            lexer = None

        formatter = Terminal256Formatter(bg='dark', style='monokai')
        text = code if lexer is None else highlight(code, lexer, formatter)
        with dcg.ChildWindow(self.C, indent=-1, auto_resize_y=True, theme=self.no_spacing):
            lines = text.split("\n")
            for line in lines:
                if line == "":
                    dcg.Spacer(self.C)
                    continue
                TextAnsi(self.C, value=line, no_wrap=True)
        return ""

    def render_thematic_break(self, element):
        #with dcg.DrawInWindow(self.C, height=8, width=10000): # TODO: fix height=1 not working
        #    dcg.DrawLine(self.C, p1 = (-100, 0), p2 = (10000, 0), color=(255, 255, 255))
        #dcg.Spacer(self.C)
        dcg.Separator(self.C)
        return ""

    def render_heading(self, element):
        level = element.level
        if self.use_auto_scale:
            # Cheat by applying a global scale only on the AutoFont attached
            scaling = self.huge_font_scale if level <= 1 else self.big_font_scale
            with dcg.Layout(self.C, font=self.default_font, scaling_factor=scaling):
                with dcg.Layout(self.C, scaling_factor=1./scaling):
                    text = self.render_children_if_not_str(element)
                    if text != "":
                        TextAnsi(self.C, wrap=self.wrap, value=text)
        else:
            font = self.huge_font if level <= 1 else self.big_font
            with dcg.Layout(self.C, font=font):
                text = self.render_children_if_not_str(element)
                if text != "":
                    TextAnsi(self.C, wrap=self.wrap, value=text)
        return ""

    def render_blank_line(self, element):
        dcg.Spacer(self.C)
        return ""

    def render_emphasis(self, element) -> str:
        return make_color(make_italic(self.render_children_if_not_str(element)), color="green")

    def render_strong_emphasis(self, element) -> str:
        return make_color(make_bold_italic(self.render_children_if_not_str(element)), color="red")

    def render_plain_text(self, element):
        return self.render_children_if_not_str(element)

    def render_raw_text(self, element):
        # Trim spaces after a "\n"
        text = self.render_children_if_not_str(element)
        subtexts = text.split('\n')
        new_subtexts = subtexts[0:1]
        for subtext in subtexts[1:]:
            i = 0
            while i < len(subtext) and text[i] == ' ':
                i = i + 1
            new_subtexts.append(subtext[i:]) 
        # convert newline into spaces
        return " ".join(new_subtexts)

    def render_image(self, element) -> str:
        with dcg.ChildWindow(self.context, auto_resize_x=True, auto_resize_y=True):
            image_path = element.dest
            if not(os.path.exists(image_path)):
                alternate_text = self.render_children_if_not_str(element)
                dcg.Text(self.context, alternate_text)
            else:
                image_content = imageio.v3.imread(image_path)
                image_texture = dcg.Texture(self.context)
                image_texture.set_value(image_content)
                dcg.Image(self.context, texture=image_texture)
            if element.title:
                with dcg.HorizontalLayout(self.context, alignment_mode=dcg.Alignment.CENTER):
                    dcg.Text(self.context, value=element.title)
        return ""

    def render_line_break(self, element):
        if element.soft:
            return " "
        return "\n"

    def render_code_span(self, element) -> str:
        text = make_bold(self.render_children_if_not_str(element))
        if True:#hasattr(dcg, text) or '_' in text:
            text = make_color(text, color="cyan")
        return text


class DrawMap(dcg.DrawInPlot):
    """
    A map drawing class that handles tiled loading and display of map data
    """
    def __init__(self, C, zoom_levels=(12, 14, 16), tile_size=256, max_tiles_per_level=100, **kwargs):
        super().__init__(C, **kwargs)
        
        # Map configuration
        self.zoom_levels = sorted(zoom_levels)  # Low to high resolution
        self.tile_size = tile_size
        self.max_tiles_per_level = max_tiles_per_level
        
        # Create tile containers for each zoom level
        self.tile_drawers = {
            zoom: DrawTiledImage(C) 
            for zoom in zoom_levels
        }
        
        # Tile management
        self.tile_cache= {
            zoom: {} for zoom in zoom_levels
        }
        self.loading_tiles = set()

        plot = self.parent
        if not isinstance(plot, dcg.Plot):
            raise ValueError("DrawMap must be a child of a Plot")

        plot.handlers += [
            dcg.AxesResizeHandler(C, axes=self.axes, callback=self._on_viewport_change),
        ]
        
        # Start worker thread for loading tiles
        self.running = True
        self.load_queue = []
        self.worker_thread = threading.Thread(target=self._tile_loader_worker)
        self.worker_thread.daemon = True
        self.worker_thread.start()

    def _get_tile_coords(self, zoom: int, lat: float, lon: float) -> Tuple[int, int]:
        """Convert lat/lon to tile coordinates"""
        lat_rad = math.radians(lat)
        n = 2.0 ** zoom
        xtile = int((lon + 180.0) / 360.0 * n)
        ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return (xtile, ytile)

    def _get_tile_bounds(self, zoom: int, x: int, y: int) -> Tuple[float, float, float, float]:
        """Get tile bounds in lat/lon"""
        n = 2.0 ** zoom
        lon_deg = x / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
        lat_deg = math.degrees(lat_rad)
        return (
            lat_deg,
            lon_deg,
            math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n)))),
            (x + 1) / n * 360.0 - 180.0
        )

    def _on_viewport_change(self, sender, target, data):
        """Handle viewport changes and update visible tiles"""
        ((min_lat, max_lat, lat_resolution), (min_lon, max_lon, lon_resolution)) = data
        
        # Queue tiles for loading at each zoom level
        for zoom in self.zoom_levels:
            min_tile_x, min_tile_y = self._get_tile_coords(zoom, min_lat, min_lon)
            max_tile_x, max_tile_y = self._get_tile_coords(zoom, max_lat, max_lon)
            
            for x in range(min_tile_x, max_tile_x + 1):
                for y in range(min_tile_y, max_tile_y + 1):
                    tile_key = (x, y)
                    if tile_key not in self.tile_cache[zoom] and tile_key not in self.loading_tiles:
                        self.load_queue.append((zoom, x, y))
                        self.loading_tiles.add(tile_key)

    def _load_tile(self, zoom: int, x: int, y: int):
        """Load a single tile from the map service"""
        try:
            # Using OpenStreetMap for this example
            url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
            response = requests.get(url)
            response.raise_for_status()
            
            # Convert to numpy array
            img = Image.open(io.BytesIO(response.content))
            tile_data = np.array(img)
            
            # Get tile bounds
            bounds = self._get_tile_bounds(zoom, x, y)
            
            # Add tile to drawer
            uuid = self.tile_drawers[zoom].add_tile(
                tile_data,
                (bounds[1], bounds[0]),  # lon, lat
                (bounds[3], bounds[2])   # lon, lat
            )
            
            # Update cache
            self.tile_cache[zoom][(x, y)] = uuid
            
        except Exception as e:
            print(f"Error loading tile {zoom}/{x}/{y}: {e}")
        finally:
            self.loading_tiles.remove((x, y))

    def _tile_loader_worker(self):
        """Worker thread for loading tiles"""
        while self.running:
            if self.load_queue:
                zoom, x, y = self.load_queue.pop(0)
                self._load_tile(zoom, x, y)
            else:
                time.sleep(0.1)

    def cleanup(self):
        """Cleanup resources"""
        self.running = False
        if self.worker_thread.is_alive():
            self.worker_thread.join()
        
        # Cleanup tile drawers
        for drawer in self.tile_drawers.values():
            drawer.cleanup()

    def draw(self):
        """Draw the map"""
        # Draw tiles from lowest to highest resolution
        for zoom in self.zoom_levels:
            self.tile_drawers[zoom].draw()

######## Slide utilities (will probably one day get in dearcygui.utils)


mono_font = None

class Slide(dcg.ChildWindow):
    """A single slide in the presentation.
    
    This class represents a single slide within a slideshow presentation. It inherits from
    dcg.ChildWindow to provide a contained area for slide content.
    
    Attributes:
        border (bool): Whether to show a border around the slide (default: False)
        no_scrollbar (bool): Disables scrollbars (default: True)
        width (float): Width of the slide (-1e-3 means fill available width)
        height (float): Height of the slide (-1e-3 means fill available height)
        title (str): Title of the slide, displayed in the menubar
        indent (int): Left indentation for slide content (default: 30)
    
    Example:
        ```python
        with Slide(C, title="My First Slide"):
            dcg.Text(C, value="Hello World!")
        ```
    """
    
    def __init__(self, C: dcg.Context, title="", **kwargs):
        self.border = False
        self.no_scrollbar = True
        self.width = -1e-3
        self.height = -1e-3
        self.title = title
        self.indent = 30
        self.no_scroll_with_mouse = True
        super().__init__(C, show=False, **kwargs)

class CenteredContent(dcg.ChildWindow):
    """A slide component that centers its content within the parent window.
    
    This container automatically centers its contents both horizontally and vertically
    within its parent container. It can be configured to center only vertically or
    horizontally if desired.
    
    Args:
        vertical_only (bool): Only center content vertically (default: False)
        horizontal_only (bool): Only center content horizontally (default: False)
        **kwargs: Additional arguments passed to dcg.ChildWindow
    
    Note:
        vertical_only and horizontal_only cannot both be True
    
    Example:
        ```python
        with CenteredContent(C):
            dcg.Text(C, value="This text will be centered!")
            
        # For vertical-only centering
        with CenteredContent(C, vertical_only=True):
            dcg.Text(C, value="This text will be centered vertically!")
        ```
    """
    
    def __init__(self, C: dcg.Context, vertical_only=False, horizontal_only=False, **kwargs):
        assert(not (vertical_only and horizontal_only))
        if vertical_only:
            self.width = -1e-3
            self.auto_resize_x = False
            self.auto_resize_y = True
        elif horizontal_only:
            self.height = -1e-3
            self.auto_resize_x = True
            self.auto_resize_y = False
        else:
            self.auto_resize_x = True
            self.auto_resize_y = True
        self.always_auto_resize = True
        self.border = False
        self.no_scrollbar = True
        self.no_scroll_with_mouse = True
        super().__init__(C, **kwargs)

        self.handlers += \
            [dcg.RenderHandler(C, callback=self.center)]
        
    def center(self):
        """Calculate and apply centering before rendering"""
        if not self.parent:
            return
            
        # Get parent window size
        available_rect_size = self.parent.rect_size
        content_size = self.content_region_avail

        left = (available_rect_size[0] - content_size[0]) // 2
        top = (available_rect_size[1] - content_size[1]) // 2
        if not self.auto_resize_x:
            left = None
        if not self.auto_resize_y:
            top = None

        self.pos_to_parent = (left, top)
        self.context.viewport.wake()

class SlideSection(dcg.ChildWindow):
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
    
    def __init__(self, C: dcg.Context, **kwargs):
        self.no_scrollbar = True
        self.border = False
        self.no_scroll_with_mouse = True
        super().__init__(C, **kwargs)

class TwoColumnSlide(Slide):
    """A slide that automatically arranges content in two columns.
    
    This specialized slide type creates a two-column layout with optional separator.
    Content is placed in the left or right column based on the order of SlideSection
    containers within the slide.
    
    Args:
        title (str): Title of the slide
        separator (bool): Whether to show a vertical separator between columns (default: True)
        **kwargs: Additional arguments passed to Slide
    
    Example:
        ```python
        with TwoColumnSlide(C, title="Two Columns"):
            with SlideSection(C):  # Left column
                dcg.Text(C, value="Left column content")
            with SlideSection(C):  # Right column
                dcg.Text(C, value="Right column content")
        ```
    
    Note:
        - Must contain exactly two SlideSection children
        - Columns are automatically sized to fill available space
        - Content in each column can be independently scrolled if needed
    """
    
    def __init__(self, C: dcg.Context, title="", separator=True, **kwargs):
        self.show_separator = separator
        self._left: Optional[SlideSection] = None
        self._right: Optional[SlideSection] = None

        super().__init__(C, title=title, **kwargs)
        # Add handler to manage layout
        self.handlers += [dcg.RenderHandler(C, callback=self.arrange_columns)]
        
    def arrange_columns(self):
        """Arrange the two columns and add separator"""
        if not self._left or not self._right:
            # Find SubSlides if not cached
            sub_slides = [c for c in self.children if isinstance(c, SlideSection)]
            if len(sub_slides) >= 2:
                self._left, self._right = sub_slides[:2]
            else:
                return
        
        # Calculate widths
        total_width = self.rect_size[0]
        column_width = (total_width - 20) // 2  # 20px for separator space
        
        # Position columns
        self._left.width = column_width
        self._right.width = column_width
        self._left.no_scaling = True
        self._right.no_scaling = True
        self._left.pos_to_parent = (0, 0)
        self._right.pos_to_parent = (total_width - column_width, 0)
        
        # Draw separator
        """
        if self.show_separator:
            draw_list = self.get_window_draw_list()
            draw_list.add_line(
                (total_width // 2, 10),
                (total_width // 2, self.rect_size[1] - 10),
                color=(128, 128, 128, 200),
                thickness=1
            )
        """
        self.context.viewport.wake()

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
            WindowBg=(30, 30, 30),
            MenuBarBg=(40, 40, 40),
            PopupBg=(35, 35, 35),
            Border=(60, 60, 60),
            BorderShadow=(0, 0, 0, 0),
            FrameBg=(50, 50, 50),
            FrameBgHovered=(70, 70, 70),
            FrameBgActive=(90, 90, 90),
            TitleBg=(40, 40, 40),
            TitleBgActive=(50, 50, 50),
            TitleBgCollapsed=(40, 40, 40),
            ScrollbarBg=(35, 35, 35),
            ScrollbarGrab=(70, 70, 70),
            ScrollbarGrabHovered=(90, 90, 90),
            ScrollbarGrabActive=(110, 110, 110),
            CheckMark=(100, 100, 255),
            SliderGrab=(100, 100, 255),
            SliderGrabActive=(150, 150, 255),
            Button=(60, 60, 60),
            ButtonHovered=(80, 80, 80),
            ButtonActive=(100, 100, 100),
            Header=(60, 60, 70),
            HeaderHovered=(70, 70, 80),
            HeaderActive=(80, 80, 90),
            Separator=(60, 60, 60),
            SeparatorHovered=(70, 70, 70),
            SeparatorActive=(90, 90, 90),
            ResizeGrip=(60, 60, 60),
            ResizeGripHovered=(80, 80, 80),
            ResizeGripActive=(100, 100, 100),
            Tab=(50, 50, 50),
            TabHovered=(70, 70, 70),
            TabSelected=(80, 80, 80),
            TabDimmed=(40, 40, 40),
            TabDimmedSelected=(50, 50, 50),
            PlotLines=(156, 156, 156),
            PlotLinesHovered=(255, 110, 89),
            PlotHistogram=(230, 179, 0),
            PlotHistogramHovered=(255, 153, 0),
            TextSelectedBg=(0, 90, 180, 180),
            DragDropTarget=(255, 255, 0, 230),
            NavWindowingHighlight=(255, 255, 255, 179),
            NavWindowingDimBg=(51, 51, 51, 179),
            ModalWindowDimBg=(35, 35, 35, 90),
            Text=(230, 230, 230))

    def _create_light_theme(self):
        return dcg.ThemeColorImGui(self.context,
            WindowBg=(245, 245, 245),
            MenuBarBg=(235, 235, 235),
            PopupBg=(240, 240, 240),
            Border=(200, 200, 200),
            BorderShadow=(0, 0, 0, 0),
            FrameBg=(235, 235, 235),
            FrameBgHovered=(225, 225, 225),
            FrameBgActive=(215, 215, 215),
            TitleBg=(235, 235, 235),
            TitleBgActive=(225, 225, 225),
            TitleBgCollapsed=(235, 235, 235),
            ScrollbarBg=(240, 240, 240),
            ScrollbarGrab=(190, 190, 190),
            ScrollbarGrabHovered=(170, 170, 170),
            ScrollbarGrabActive=(150, 150, 150),
            CheckMark=(100, 100, 255),
            SliderGrab=(100, 100, 255),
            SliderGrabActive=(80, 80, 255),
            Button=(225, 225, 225),
            ButtonHovered=(215, 215, 215),
            ButtonActive=(200, 200, 200),
            Header=(220, 220, 230),
            HeaderHovered=(210, 210, 220),
            HeaderActive=(200, 200, 210),
            Separator=(200, 200, 200),
            SeparatorHovered=(190, 190, 190),
            SeparatorActive=(180, 180, 180),
            ResizeGrip=(200, 200, 200),
            ResizeGripHovered=(190, 190, 190),
            ResizeGripActive=(180, 180, 180),
            Tab=(220, 220, 220),
            TabHovered=(210, 210, 210),
            TabSelected=(200, 200, 200),
            TabDimmed=(230, 230, 230),
            TabDimmedSelected=(220, 220, 220),
            PlotLines=(100, 100, 100),
            PlotLinesHovered=(255, 110, 89),
            PlotHistogram=(230, 179, 0),
            PlotHistogramHovered=(255, 153, 0),
            TextSelectedBg=(173, 214, 255),
            DragDropTarget=(255, 255, 0, 230),
            NavWindowingHighlight=(0, 0, 0, 179),
            NavWindowingDimBg=(204, 204, 204, 179),
            ModalWindowDimBg=(204, 204, 204, 90),
            Text=(30, 30, 30))

    def _create_sepia_theme(self):
        return dcg.ThemeColorImGui(self.context,
            WindowBg=(251, 240, 217),
            MenuBarBg=(242, 229, 201),
            PopupBg=(251, 240, 217),
            Border=(200, 186, 157),
            BorderShadow=(0, 0, 0, 0),
            FrameBg=(242, 229, 201),
            FrameBgHovered=(236, 221, 188),
            FrameBgActive=(229, 212, 175),
            TitleBg=(242, 229, 201),
            TitleBgActive=(236, 221, 188),
            TitleBgCollapsed=(242, 229, 201),
            ScrollbarBg=(242, 229, 201),
            ScrollbarGrab=(220, 201, 159),
            ScrollbarGrabHovered=(200, 186, 157),
            ScrollbarGrabActive=(180, 166, 137),
            CheckMark=(180, 166, 137),
            SliderGrab=(200, 186, 157),
            SliderGrabActive=(180, 166, 137),
            Button=(236, 221, 188),
            ButtonHovered=(229, 212, 175),
            ButtonActive=(220, 201, 159),
            Header=(236, 221, 188),
            HeaderHovered=(229, 212, 175),
            HeaderActive=(220, 201, 159),
            Separator=(200, 186, 157),
            SeparatorHovered=(180, 166, 137),
            SeparatorActive=(160, 146, 117),
            ResizeGrip=(220, 201, 159),
            ResizeGripHovered=(200, 186, 157),
            ResizeGripActive=(180, 166, 137),
            Tab=(236, 221, 188),
            TabHovered=(229, 212, 175),
            TabSelected=(220, 201, 159),
            TabDimmed=(242, 229, 201),
            TabDimmedSelected=(236, 221, 188),
            PlotLines=(160, 146, 117),
            PlotLinesHovered=(180, 166, 137),
            PlotHistogram=(200, 186, 157),
            PlotHistogramHovered=(180, 166, 137),
            TextSelectedBg=(200, 186, 157, 180),
            DragDropTarget=(180, 166, 137, 230),
            NavWindowingHighlight=(200, 186, 157, 179),
            NavWindowingDimBg=(251, 240, 217, 179),
            ModalWindowDimBg=(251, 240, 217, 90),
            Text=(101, 80, 40))

    def _create_nord_theme(self):
        return dcg.ThemeColorImGui(self.context,
            WindowBg=(46, 52, 64),
            MenuBarBg=(59, 66, 82),
            PopupBg=(46, 52, 64),
            Border=(76, 86, 106),
            BorderShadow=(0, 0, 0, 0),
            FrameBg=(67, 76, 94),
            FrameBgHovered=(76, 86, 106),
            FrameBgActive=(86, 97, 119),
            TitleBg=(59, 66, 82),
            TitleBgActive=(67, 76, 94),
            TitleBgCollapsed=(59, 66, 82),
            ScrollbarBg=(46, 52, 64),
            ScrollbarGrab=(76, 86, 106),
            ScrollbarGrabHovered=(86, 97, 119),
            ScrollbarGrabActive=(93, 104, 126),
            CheckMark=(136, 192, 208),
            SliderGrab=(129, 161, 193),
            SliderGrabActive=(136, 192, 208),
            Button=(67, 76, 94),
            ButtonHovered=(76, 86, 106),
            ButtonActive=(86, 97, 119),
            Header=(67, 76, 94),
            HeaderHovered=(76, 86, 106),
            HeaderActive=(86, 97, 119),
            Separator=(76, 86, 106),
            SeparatorHovered=(86, 97, 119),
            SeparatorActive=(93, 104, 126),
            ResizeGrip=(76, 86, 106),
            ResizeGripHovered=(86, 97, 119),
            ResizeGripActive=(93, 104, 126),
            Tab=(67, 76, 94),
            TabHovered=(76, 86, 106),
            TabSelected=(86, 97, 119),
            TabDimmed=(59, 66, 82),
            TabDimmedSelected=(67, 76, 94),
            PlotLines=(136, 192, 208),
            PlotLinesHovered=(143, 188, 187),
            PlotHistogram=(129, 161, 193),
            PlotHistogramHovered=(136, 192, 208),
            TextSelectedBg=(76, 86, 106, 180),
            DragDropTarget=(136, 192, 208, 230),
            NavWindowingHighlight=(129, 161, 193, 179),
            NavWindowingDimBg=(46, 52, 64, 179),
            ModalWindowDimBg=(46, 52, 64, 90),
            Text=(236, 239, 244))

    def _create_dracula_theme(self):
        return dcg.ThemeColorImGui(self.context,
            WindowBg=(40, 42, 54),
            MenuBarBg=(68, 71, 90),
            PopupBg=(40, 42, 54),
            Border=(98, 114, 164),
            BorderShadow=(0, 0, 0, 0),
            FrameBg=(68, 71, 90),
            FrameBgHovered=(78, 82, 104),
            FrameBgActive=(88, 91, 112),
            TitleBg=(68, 71, 90),
            TitleBgActive=(78, 82, 104),
            TitleBgCollapsed=(68, 71, 90),
            ScrollbarBg=(40, 42, 54),
            ScrollbarGrab=(68, 71, 90),
            ScrollbarGrabHovered=(78, 82, 104),
            ScrollbarGrabActive=(88, 91, 112),
            CheckMark=(189, 147, 249),
            SliderGrab=(189, 147, 249),
            SliderGrabActive=(255, 121, 198),
            Button=(68, 71, 90),
            ButtonHovered=(78, 82, 104),
            ButtonActive=(88, 91, 112),
            Header=(68, 71, 90),
            HeaderHovered=(78, 82, 104),
            HeaderActive=(88, 91, 112),
            Separator=(98, 114, 164),
            SeparatorHovered=(108, 124, 174),
            SeparatorActive=(118, 134, 184),
            ResizeGrip=(68, 71, 90),
            ResizeGripHovered=(78, 82, 104),
            ResizeGripActive=(88, 91, 112),
            Tab=(68, 71, 90),
            TabHovered=(78, 82, 104),
            TabSelected=(88, 91, 112),
            TabDimmed=(40, 42, 54),
            TabDimmedSelected=(68, 71, 90),
            PlotLines=(248, 248, 242),
            PlotLinesHovered=(255, 121, 198),
            PlotHistogram=(189, 147, 249),
            PlotHistogramHovered=(255, 121, 198),
            TextSelectedBg=(98, 114, 164, 180),
            DragDropTarget=(189, 147, 249, 230),
            NavWindowingHighlight=(189, 147, 249, 179),
            NavWindowingDimBg=(40, 42, 54, 179),
            ModalWindowDimBg=(40, 42, 54, 90),
            Text=(248, 248, 242))

    def _create_solarized_dark_theme(self):
        return dcg.ThemeColorImGui(self.context,
            WindowBg=(0, 43, 54),
            MenuBarBg=(7, 54, 66),
            PopupBg=(0, 43, 54),
            Border=(88, 110, 117),
            BorderShadow=(0, 0, 0, 0),
            FrameBg=(7, 54, 66),
            FrameBgHovered=(23, 66, 77),
            FrameBgActive=(32, 77, 87),
            TitleBg=(7, 54, 66),
            TitleBgActive=(23, 66, 77),
            TitleBgCollapsed=(7, 54, 66),
            ScrollbarBg=(0, 43, 54),
            ScrollbarGrab=(88, 110, 117),
            ScrollbarGrabHovered=(101, 123, 131),
            ScrollbarGrabActive=(131, 148, 150),
            CheckMark=(181, 137, 0),
            SliderGrab=(133, 153, 0),
            SliderGrabActive=(181, 137, 0),
            Button=(7, 54, 66),
            ButtonHovered=(23, 66, 77),
            ButtonActive=(32, 77, 87),
            Header=(7, 54, 66),
            HeaderHovered=(23, 66, 77),
            HeaderActive=(32, 77, 87),
            Separator=(88, 110, 117),
            SeparatorHovered=(101, 123, 131),
            SeparatorActive=(131, 148, 150),
            ResizeGrip=(88, 110, 117),
            ResizeGripHovered=(101, 123, 131),
            ResizeGripActive=(131, 148, 150),
            Tab=(7, 54, 66),
            TabHovered=(23, 66, 77),
            TabSelected=(32, 77, 87),
            TabDimmed=(0, 43, 54),
            TabDimmedSelected=(7, 54, 66),
            PlotLines=(147, 161, 161),
            PlotLinesHovered=(181, 137, 0),
            PlotHistogram=(133, 153, 0),
            PlotHistogramHovered=(181, 137, 0),
            TextSelectedBg=(88, 110, 117, 180),
            DragDropTarget=(133, 153, 0, 230),
            NavWindowingHighlight=(133, 153, 0, 179),
            NavWindowingDimBg=(0, 43, 54, 179),
            ModalWindowDimBg=(0, 43, 54, 90),
            Text=(147, 161, 161))

    def _create_ocean_theme(self):
        return dcg.ThemeColorImGui(self.context,
            WindowBg=(28, 45, 65),
            MenuBarBg=(36, 55, 77),
            PopupBg=(28, 45, 65),
            Border=(52, 74, 100),
            BorderShadow=(0, 0, 0, 0),
            FrameBg=(36, 55, 77),
            FrameBgHovered=(52, 74, 100),
            FrameBgActive=(64, 89, 119),
            TitleBg=(36, 55, 77),
            TitleBgActive=(52, 74, 100),
            TitleBgCollapsed=(36, 55, 77),
            ScrollbarBg=(28, 45, 65),
            ScrollbarGrab=(52, 74, 100),
            ScrollbarGrabHovered=(64, 89, 119),
            ScrollbarGrabActive=(77, 106, 141),
            CheckMark=(103, 183, 255),
            SliderGrab=(71, 161, 241),
            SliderGrabActive=(103, 183, 255),
            Button=(36, 55, 77),
            ButtonHovered=(52, 74, 100),
            ButtonActive=(64, 89, 119),
            Header=(36, 55, 77),
            HeaderHovered=(52, 74, 100),
            HeaderActive=(64, 89, 119),
            Separator=(52, 74, 100),
            SeparatorHovered=(64, 89, 119),
            SeparatorActive=(77, 106, 141),
            ResizeGrip=(52, 74, 100),
            ResizeGripHovered=(64, 89, 119),
            ResizeGripActive=(77, 106, 141),
            Tab=(36, 55, 77),
            TabHovered=(52, 74, 100),
            TabSelected=(64, 89, 119),
            TabDimmed=(28, 45, 65),
            TabDimmedSelected=(36, 55, 77),
            PlotLines=(154, 206, 255),
            PlotLinesHovered=(103, 183, 255),
            PlotHistogram=(71, 161, 241),
            PlotHistogramHovered=(103, 183, 255),
            TextSelectedBg=(52, 74, 100, 180),
            DragDropTarget=(71, 161, 241, 230),
            NavWindowingHighlight=(71, 161, 241, 179),
            NavWindowingDimBg=(28, 45, 65, 179),
            ModalWindowDimBg=(28, 45, 65, 90),
            Text=(192, 215, 235))

    def _create_forest_theme(self):
        return dcg.ThemeColorImGui(self.context,
            WindowBg=(35, 45, 35),
            MenuBarBg=(45, 55, 45),
            PopupBg=(35, 45, 35),
            Border=(65, 85, 65),
            BorderShadow=(0, 0, 0, 0),
            FrameBg=(45, 55, 45),
            FrameBgHovered=(55, 75, 55),
            FrameBgActive=(65, 85, 65),
            TitleBg=(45, 55, 45),
            TitleBgActive=(55, 75, 55),
            TitleBgCollapsed=(45, 55, 45),
            ScrollbarBg=(35, 45, 35),
            ScrollbarGrab=(65, 85, 65),
            ScrollbarGrabHovered=(75, 95, 75),
            ScrollbarGrabActive=(85, 105, 85),
            CheckMark=(141, 197, 62),
            SliderGrab=(126, 179, 51),
            SliderGrabActive=(141, 197, 62),
            Button=(45, 55, 45),
            ButtonHovered=(55, 75, 55),
            ButtonActive=(65, 85, 65),
            Header=(45, 55, 45),
            HeaderHovered=(55, 75, 55),
            HeaderActive=(65, 85, 65),
            Separator=(65, 85, 65),
            SeparatorHovered=(75, 95, 75),
            SeparatorActive=(85, 105, 85),
            ResizeGrip=(65, 85, 65),
            ResizeGripHovered=(75, 95, 75),
            ResizeGripActive=(85, 105, 85),
            Tab=(45, 55, 45),
            TabHovered=(55, 75, 55),
            TabSelected=(65, 85, 65),
            TabDimmed=(35, 45, 35),
            TabDimmedSelected=(45, 55, 45),
            PlotLines=(180, 210, 120),
            PlotLinesHovered=(141, 197, 62),
            PlotHistogram=(126, 179, 51),
            PlotHistogramHovered=(141, 197, 62),
            TextSelectedBg=(65, 85, 65, 180),
            DragDropTarget=(126, 179, 51, 230),
            NavWindowingHighlight=(126, 179, 51, 179),
            NavWindowingDimBg=(35, 45, 35, 179),
            ModalWindowDimBg=(35, 45, 35, 90),
            Text=(210, 230, 190))

    def toggle(self):
        """Cycle to the next available theme"""
        self.current_index = (self.current_index + 1) % len(self.themes)
        self.current_theme = self.themes[self.current_index][1]
        self.children = [self.current_theme]
        return self.themes[self.current_index][0]  # Return theme name

def _get_unique_filename(base_name: str, ext: str) -> str:
    """Generate a unique filename by appending a number if file exists"""
    if not os.path.exists(f"{base_name}.{ext}"):
        return f"{base_name}.{ext}"
    counter = 1
    while os.path.exists(f"{base_name}_({counter}).{ext}"):
        counter += 1
    return f"{base_name}_({counter}).{ext}"

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
        global mono_font
        super().__init__(C, label=title, no_collapse=True, **kwargs)
        self.slides: List[Slide] = []
        self.current_slide: int = 0
        self.font = dcg.AutoFont(C)
        self.mono_font = dcg.AutoFont(C, 14, main_font_path="NotoSansMono-Regular.ttf")
        mono_font = self.mono_font
        self.primary = True
        self.no_move = True
        self.no_title_bar = True
        self.padding = (10, 10)
        
        # Setup dark theme
        with dcg.ThemeList(self.context) as theme:
            self.theme_color = ThemeColorVariant(self.context)
            style_imgui = \
            dcg.ThemeStyleImGui(self.context,
                WindowPadding=(10, 10),
                FramePadding=(6, 3),
                ItemSpacing=(8, 6),
                ScrollbarSize=12,
                GrabMinSize=20,
                WindowBorderSize=1,
                ChildBorderSize=1,
                FrameBorderSize=0,
                WindowRounding=0,
                FrameRounding=4,
                PopupRounding=4,
                ScrollbarRounding=4,
                GrabRounding=4)
            style_implot = dcg.ThemeStyleImPlot(self.context)

        # Since we change the scaling factor, in order to have it apply
        # to all theme values, we need to reapply the theme entirely.
        # thus we fill in the theme with the default values
        for name in [name for name in dir(style_imgui) if name[0].isupper()]:
            if name in style_imgui:
                # non default
                continue
            style_imgui[name] = style_imgui.get_default(name)
        for name in [name for name in dir(style_implot) if name[0].isupper()]:
            style_implot[name] = style_implot.get_default(name)

        self.theme = theme

        # Create theme for colored text that will contrast with background
        self.colored_text_theme = dcg.ThemeColorImGui(self.context)
        self._update_colored_text_theme()

        self._setup_menubar()
        self._setup_progress_bar()

    def _update_colored_text_theme(self):
        """Update the colored text theme based on current theme background"""
        bg_color = dcg.color_as_floats(self.theme_color.current_theme.WindowBg)
        # Simple luminance calculation
        luminance = (0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2])
        
        if luminance < 0.5:
            # Dark background - use bright yellow
            self.colored_text_theme.Text = (255, 255, 0)
        else:
            # Light background - use dark blue 
            self.colored_text_theme.Text = (0, 0, 180)

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
                    dcg.show_save_file_dialog(save_callback)

                dcg.Button(self.context, label="Quick save", callback=quick_save)
                dcg.Button(self.context, label="Save as...", callback=save_as)

        with dcg.MenuBar(self.context, parent=self):
            # Navigation buttons with icons
            dcg.Button(self.context, arrow=True, direction=dcg.ButtonDirection.LEFT,
                       callback=self.previous_slide)
            dcg.Button(self.context, arrow=True, direction=dcg.ButtonDirection.RIGHT,
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
                               clamped=True,
                               width=150,
                               callback=self._update_scale)
                
            scale_button.callbacks = open_scale_popup
                
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
            dcg.KeyPressHandler(self.context, callback=self.previous_slide, key=dcg.Key.LEFTARROW),
            dcg.KeyPressHandler(self.context, callback=self.next_slide, key=dcg.Key.RIGHTARROW),
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
                                                 width=-1e-3,
                                                 height=3)
            #dcg.Spacer(self.context, height=5)

    def _toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        self.context.viewport.fullscreen = not self.context.viewport.fullscreen

    def _toggle_theme(self):
        """Toggle between color themes"""
        theme_name = self.theme_color.toggle()
        self._theme_label.value = theme_name
        self._update_colored_text_theme()

    def _update_slide_counter(self):
        """Update the slide counter text"""
        self._slide_counter.value = f"Slide: {self.current_slide + 1}/{len(self.slides)}"

    def _update_scale(self, sender, target, value: float):
        """Update the font scale"""
        self.scaling_factor = value

    def show_slide(self, index: int):
        """Show the slide at the given index"""
        if 0 <= index < len(self.slides):
            # Hide current slide
            if self.slides[self.current_slide].show:
                self.slides[self.current_slide].show = False
            
            # Show new slide
            self.current_slide = index
            self.slides[index].show = True
            self._update_slide_counter()
            
            # Update slide title
            self._slide_title.value = dcg.font.make_bold_italic(self.slides[index].title)
            self._progress_bar.value = (index + 1) / len(self.slides)
            self._progress_bar.overlay = f"{index + 1}/{len(self.slides)}"
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
        self.show_slide(0)

    def colored_text(self, text: str, **kwargs):
        """Display colored text in the current theme"""
        return dcg.Text(self.context, value=text, theme=self.colored_text_theme, **kwargs)

    def separator(self):
        """Add a separator line"""
        return dcg.Separator(self.context)

    def readonly_code(self, text: str | dcg.SharedStr, height : float = -1e-3, width : float = -1e-3):
        """Display read-only code in the current theme"""
        if isinstance(text, str):
            text = dcg.SharedStr(self.context, value=text)
        return dcg.InputText(self.context, shareable_value=text, font=self.mono_font,
                             max_characters=len(text.value),
                             readonly=True, multiline=True,
                             height=height, width=width)

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
    
    def __init__(self, C: dcg.Context, initial_code="",
                 width=440, height=440,
                 **kwargs):
        super().__init__(C, width=width, height=height, **kwargs)
        self.no_scrollbar = True
        self.border = False
        self.no_scroll_with_mouse = True
        
        # Add code editor
        self.editor = dcg.InputText(C, parent=self,
            width=-1e-3,
            height=-34,
            multiline=True,
            tab_input=True,
            font=mono_font,
            value=initial_code)
        with dcg.HorizontalLayout(C, parent=self,
                                  alignment_mode=dcg.Alignment.RIGHT):
            dcg.Button(C, label="Run", callback=self._run_code)
            self.status = dcg.Text(C, value="")

        self._last_log = ""
        self._last_output = None
        self.output_text = None
        self.image_display = None
        self.plot_display = None
        
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
    
    def __init__(self, C: dcg.Context,
                 texture=None,
                 **kwargs):
        super().__init__(C, **kwargs)
        
        # Configure plot to be minimal
        self.no_mouse_pos = True
        self.no_menus = True
        self.no_frame = True
        self.no_title = True
        self.no_legend = True
        self.equal_aspects = True
        self.theme = dcg.ThemeStyleImPlot(C, PlotBorderSize=0)
        
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
        with dcg.DrawInPlot(C, parent=self):
            self._image = dcg.DrawImage(C, pmin=(0, 0), pmax=(1, 1))
        
        if texture is not None:
            self.texture = texture

        self.handlers += [
            dcg.ClickedHandler(C, callback=self._on_right_click, button=dcg.MouseButton.RIGHT),
        ]
            
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
                        width=self.context.viewport.width,
                        height=self.context.viewport.height,
                        no_scaling=True):
            EmbeddedImage(self.context, texture=self.texture, width=-1, height=-1)


def export_slideshow(target : str, slideshow : SlideShow):
    """Export the slideshow to a pdf file.
    """
    try:
        import imageio
        from reportlab.pdfgen import canvas
        from reportlab.lib.utils import ImageReader
    except ImportError:
        raise ImportError("Exporting slides requires imageio and reportlab")
    images = []
    slideshow.context.viewport.retrieve_framebuffer = True
    current_slide_num = slideshow.current_slide
    slideshow.start()
    for _ in range(len(slideshow.slides)):
        # force convergence of the content
        # ChildWindows can take a few frames to converge,
        # and we need center callbacks to converge as well
        try:
            slideshow.context.viewport.render_frame(can_skip_presenting=True)
            slideshow.context.viewport.wake()
            slideshow.context.viewport.render_frame(can_skip_presenting=True)
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


####### The presentation itself

def slideshow_dearcygui():
    C = dcg.Context()
    slideshow = SlideShow(C, title="DearCyGui for Scientific Applications")
    slideshow.scaling_factor = 1.6
    
    with slideshow:
        # First few slides remain the same until the Interactive Example
        # Title slide
        with Slide(C, title="DearCyGui", font=slideshow.font, scaling_factor=2.):
            with CenteredContent(C):
                with dcg.HorizontalLayout(C, alignment_mode=dcg.Alignment.CENTER):
                    slideshow.colored_text("DearCyGui", font=slideshow.font, scaling_factor=2.)
                with dcg.HorizontalLayout(C, alignment_mode=dcg.Alignment.CENTER):
                    dcg.Text(C, value="A fast and Python-centric GUI Framework")
                slideshow.separator()
                dcg.Text(C, value="Building Interactive Image Processing Applications")
                with dcg.HorizontalLayout(C, alignment_mode=dcg.Alignment.CENTER):
                    dcg.Text(C, value=dcg.font.make_italic("By Axel Davy"))

        # Motivation slide
        with TwoColumnSlide(C, title="Why Do We Need a GUI?", font=slideshow.font, scaling_factor=0.9):
            with SlideSection(C):
                if True:
                    text = \
"""
**Problement statement**

Workflow 1:
- Running portions of code in notebook/console with specific inputs
- visualize the data (via print, vpv, matplotlib, etc)

**Back and forth needed !**

Workflow 2:
- Looking at a lot of images with vpv
- Relaunch vpv after having run an algorithm that couldn't be expressed in plambda

**I only we had embedded Python**

Workflow 3:
- You use vpv and you wish you could draw without using an svg and interact with the data

**Vpv plugin are useful but that's not Python**
"""
                MarkDownText(C, value=text)
            with SlideSection(C):
                if True:
                    text = \
"""
*DearCyGui* is a home-made library written in *Cython*.

It can:
- Make UIs (buttons, sliders, etc)
- Draw plots
- Add interactable items to the plot
- Enable to create custom objects
- Show data from any other python library

A lot of effort has been spent for:
- Good documentation
- Typing, Autocompletion
- Performance
"""
                    MarkDownText(C, value=text)
        # GUI Frameworks Comparison
        with Slide(C, title="Available GUI Solutions"):
            with dcg.HorizontalLayout(C, positions=[0, 0.55]): # 0%, 55% of width
                with dcg.VerticalLayout(C):
                    slideshow.colored_text("Traditional GUI Frameworks")
                    dcg.Text(C, value="Qt / GTK / wxWidgets:")
                    dcg.Text(C, value="+ Mature and feature-rich")
                    dcg.Text(C, value="- Complex Python bindings")
                    dcg.Text(C, value="- Not designed for scientific computing")
                with dcg.VerticalLayout(C):
                    slideshow.colored_text("DearCygui")
                    dcg.Text(C, value="+ Developped by someone you can talk to")
                    dcg.Text(C, value="- Not mature")
                    dcg.Text(C, value="+ Well integrated with Python")
                    dcg.Text(C, value="+ Developped for interactive data")
            slideshow.separator()
            with dcg.HorizontalLayout(C, positions=[0, 0.55]):
                with dcg.VerticalLayout(C):
                    slideshow.colored_text("Scientific Visualization")
                    dcg.Text(C, value="Matplotlib / Plotly / VTK:")
                    dcg.Text(C, value="+ Great for static visualization")
                    dcg.Text(C, value="+ Scientific computing focus")
                    dcg.Text(C, value="- Limited interactive capabilities")
                    dcg.Text(C, value="- Not suited for full applications")
                with dcg.VerticalLayout(C):
                    slideshow.colored_text("DearCygui")
                    dcg.Text(C, value="Can show results from Matplotlib, etc")
                    dcg.Text(C, value="+ Has core tools for data visualization")
                    dcg.Text(C, value="+ Developed for me and for you")
                    dcg.Text(C, value="+ Interactive: Python callbacks")
                    dcg.Text(C, value="+ Suited for a pro application")

        # IMGUI vs RMGUI
        with Slide(C, title="Understanding RMGUI vs IMGUI"):
            with dcg.HorizontalLayout(C, positions=[0, 0.55]):
                with dcg.VerticalLayout(C):
                    slideshow.colored_text("Traditional Retained Mode GUI (RMGUI):")
                    text = \
'''
- Your code create items with the library.
- The library handles item rendering and management.
- For complex interactions, a lot of boilerplate is needed.

*DearCygui* provides a *Retained Mode* API,

but uses *Immediate Mode* underneath.
'''
                    MarkDownText(C, value=text)  
                with dcg.VerticalLayout(C):
                    slideshow.colored_text("Immediate Mode GUI (IMGUI):")
                    text = \
"""
- Your code manages itself its items
- EVERY frame, you call the library to render your items
- The library is optimized to render as fast as possible
- The library may be slower in cases where having a precomputed result would be useful.
"""
                    MarkDownText(C, value=text)
            slideshow.separator()
            with dcg.HorizontalLayout(C, positions=[0, 0.55]):
                with dcg.VerticalLayout(C):
                    with dcg.HorizontalLayout(C, positions=[0, 0.25]):
                        with dcg.VerticalLayout(C):
                            slideshow.colored_text("Your code:")
                            dcg.Text(C, value="Create items", bullet=True)
                            dcg.Text(C, value="Handle events", bullet=True)
                            dcg.Text(C, value="Take action", bullet=True)
                        with dcg.VerticalLayout(C):
                            slideshow.colored_text("Library:")
                            dcg.Text(C, value="Render items", bullet=True)
                            dcg.Text(C, value="Trigger events", bullet=True)
                            dcg.Text(C, value="Fill items states", bullet=True)    
                with dcg.VerticalLayout(C):
                    with dcg.HorizontalLayout(C, positions=[0, 0.5]):
                        with dcg.VerticalLayout(C):
                            slideshow.colored_text("Your code:")
                            dcg.Text(C, value="Create items", bullet=True)
                            dcg.Text(C, value="Check events", bullet=True)
                            dcg.Text(C, value="Render items", bullet=True)
                        with dcg.VerticalLayout(C):
                            slideshow.colored_text("Library (Toolbox):")
                            dcg.Text(C, value="check events", bullet=True)
                            dcg.Text(C, value="render items", bullet=True)
                            dcg.Text(C, value="organize items", bullet=True)

        # Performance Benchmark Slide
        with TwoColumnSlide(C, title="Performance Matters"):
            with SlideSection(C):
                slideshow.colored_text("Array Operations Benchmark")
                slideshow.separator()
                
                # Interactive benchmark code
                code_editor = InteractiveCode(C, initial_code="""import numpy as np
import time

# Regular Python
def fill_array_python(size):
    arr = np.zeros(size)
    for i in range(size):
        arr[i] = i
    return arr

t1 = time.time()
fill_array_python(1000000)
t2 = time.time()
print(f"Python: {1000*(t2-t1):.2f}ms")

t1 = time.time()
np.arange(1000000)
t2 = time.time()
print(f"Numpy: {1000*(t2-t1):.2f}ms")
                                              
t1 = time.time()
np.arange(1000000, dtype=object)
t2 = time.time()
print(f"Numpy (but Python objects): {1000*(t2-t1):.2f}ms")
                                              
""", width=-1e-3, height=-1e-3)
            with SlideSection(C):
                with CenteredContent(C, vertical_only=True):
                    dcg.Text(C, value="Some Python IMGUI Libraries exist:")
                    dcg.Text(C, value="Direct C++ API mapping - error prone", bullet=True)
                    dcg.Text(C, value="Unsafe API calls with limited type checking", bullet=True)
                    dcg.Text(C, value="High Python overhead from frequent calls", bullet=True)
                    slideshow.separator()
                    dcg.Text(C, value="Python is a scripting API. Not adapted for many calls", bullet=True, wrap=0)
                    with dcg.VerticalLayout(C, indent=80, theme=slideshow.colored_text_theme):
                        code_editor.display_result()
                    dcg.Text(C, value="DearCyGui is Cython (C++) and Python friendly")
                    dcg.Text(C, value="Type-safe Python API", indent=40, bullet=True)
                    dcg.Text(C, value="Optimized Cython implementation", indent=40, bullet=True)
                    dcg.Text(C, value="Better error handling and debugging", indent=40, bullet=True)

        with Slide(C, title="Can it be fast?"):
            text = \
"""
This laptop, on battery, can:
- Render hello world at 2441 fps
- Render 20000 buttons at 55 fps
- Render 20000 anti-aliased lines at 134 fps
- Create and configure a simple button in Python at 77K fps
- Basically you can create and configure 1000-10000 items per frame
- Does not use CPU/GPU if content doesn't need to change.
"""
            MarkDownText(C, value=text)
            
            # Add CPU usage monitoring
            import psutil
            import threading
            process = psutil.Process()

            cpu_usage = dcg.ProgressBar(C, value=0.0, overlay="0% CPU Usage", width=-1e-3)
            fps_bar = dcg.ProgressBar(C, value=0.0, overlay="0 FPS", width=-1e-3)
            # Update every second in a thread
            def update_stats(C=C, cpu_usage=cpu_usage, fps_bar=fps_bar):
                last_frame_count = C.viewport.metrics["frame_count"]
                while C.running:
                    # Update CPU usage
                    cpu_percent = process.cpu_percent()
                    cpu_usage.value = cpu_percent / 100.
                    cpu_usage.overlay = f"{cpu_percent:.2f}% CPU Usage"
                    
                    # Update FPS
                    current_frame_count = C.viewport.metrics["frame_count"]
                    fps = current_frame_count - last_frame_count
                    fps_bar.value = min(fps / 100., 1.0)  # Normalize to 100fps max
                    fps_bar.overlay = f"{fps} FPS"
                    last_frame_count = current_frame_count
                    
                    C.viewport.wake()
                    time.sleep(1)
            threading.Thread(target=update_stats, daemon=True).start()

            # Add controls for random button animation
            class RandomButtonsDemo:
                def __init__(self, C, width=1000, max_height=500):
                    self.C = C
                    self.width = width
                    self.max_height = max_height
                    self.running = False
                    self.num_buttons = 1000
                    self.last_frame_count = 0
                    self.rendered_frame = 0
                    
                    # Create controls
                    with dcg.HorizontalLayout(C):
                        self.toggle_button = dcg.Button(C, label="Start",
                            callback=self.toggle_animation)
                        dcg.Slider(C, label="Buttons", 
                            value=self.num_buttons,
                            logarithmic=True,
                            min_value=10,
                            max_value=100000,
                            callback=self.update_count)
                        
                    # Container for buttons
                    self.container = dcg.ChildWindow(C, 
                        width=-1,
                        height=-1,
                        no_scrollbar=True)
                    
                    # Create initial buttons
                    self.buttons = [
                        dcg.Button(C, label=f"Btn {i}", 
                            parent=self.container,
                            width=60,
                            height=25)
                        for i in range(self.num_buttons)
                    ]
                    
                    # Add frame handler to update positions
                    self.container.handlers += [
                        dcg.RenderHandler(C, callback=self.update_positions)
                    ]

                def toggle_animation(self):
                    self.running = not self.running
                    self.last_frame_count = self.C.viewport.metrics["frame_count"]
                    self.rendered_frame = 0
                    self.toggle_button.label = "Stop" if self.running else "Start"
                    
                def update_count(self, sender, target, value):
                    self.num_buttons = int(value)
                    # Update button list
                    while len(self.buttons) < self.num_buttons:
                        self.buttons.append(
                            dcg.Button(self.C, 
                                label=f"Btn {len(self.buttons)}", 
                                parent=self.container,
                                width=60,
                                height=25)
                        )
                    while len(self.buttons) > self.num_buttons:
                        self.buttons.pop().detach_item()
                        
                def update_positions(self):
                    if not self.running:
                        return
                    cur_frame_count = self.C.viewport.metrics["frame_count"]
                    # If we are late, skip the frame
                    if cur_frame_count - (self.last_frame_count + self.rendered_frame) > 2:
                        self.rendered_frame += 1
                        C.viewport.wake()
                        return
                    # Get container dimensions
                    w = self.container.rect_size[0] - 60  # Account for button width
                    h = self.container.rect_size[1] - 25  # Account for button height
                    num_buttons = len(self.buttons)
                    coords = np.random.randint(0, max(1, w), (num_buttons, 2))
                    # Update all button positions randomly
                    for (btn, coord) in zip(self.buttons, coords):
                        btn.pos_to_parent = coord
                    self.rendered_frame += 1
                    C.viewport.wake()

            # Create demo instance
            RandomButtonsDemo(C)


        with TwoColumnSlide(C, title="Building a minimal application"):
            code_demo = dcg.SharedStr(C, " "*1024)
            def show_context_code():
                code_demo.value = \
"""
import dearcygui as dcg

C = dcg.Context()
C.viewport.initialize()

# Build your UI here

# Render thread
while C.running:
    C.viewport.render_frame()
"""
            def show_image_code():
                code_demo.value = \
"""
import dearcygui as dcg
import numpy as np

C = dcg.Context()
C.viewport.initialize()
with dcg.Window(C, primary=True):
    with dcg.Plot(C, width=-1, height=-1):
        with dcg.DrawInPlot(C):
            image = imageio.imread("lenapashm.png")
            texture = dcg.Texture(C, image)
            dcg.DrawImage(C,
                          texture=texture,
                          pmin=(0, 1),
                          pmax=(1, 0))
while C.running:
    C.viewport.render_frame()
"""
            def show_controls_code():
                code_demo.value = \
"""
[...]
def denoise_image():
    [...]
def update_strength(sender, target, value):
    strength = value
    [...]
[...]
    with dcg.MenuBar(C):
        dcg.Button(C, label="Denoise",
                   callback=denoise_image)
        dcg.Slider(C, label="Strength",
                   value=0.5, min_value=0.0,
                   max_value=1.0, width=50,
                   callback=update_strength)
"""
            def show_event_code():
                code_demo.value = \
"""
[...]
my_plot.handlers = [
    dcg.ClickedHandler(C, callback=on_click),
    dcg.GotHoverHandler(C, callback=on_hover),
    dcg.DragHandler(C, callback=on_drag),
    dcg.KeyPressHandler(C, 
        key=dcg.Key.UPARROW,
        callback=on_keypress),
]
"""
            with SlideSection(C):
                dcg.Text(C, value="1. Create a ", no_newline=True)
                dcg.Button(C, label="Context", small=True, callback=show_context_code)
                dcg.Text(C, value="2. Display an ", no_newline=True)
                dcg.Button(C, label="Image", small=True, callback=show_image_code)
                dcg.Text(C, value="3. Add ", no_newline=True)
                dcg.Button(C, label="Controls", small=True, callback=show_controls_code)
                dcg.Text(C, value="4. Handle ", no_newline=True)
                dcg.Button(C, label="Events", small=True, callback=show_event_code)
                with dcg.ChildWindow(C, width=400, height=400):
                    dcg.Button
                    with dcg.MenuBar(C):
                        dcg.Button(C,
                                   label="Denoise")
                        dcg.Slider(C,
                                   label="Strength",
                                   value=0.5, min_value=0.0,
                                   max_value=1.0, width=50)
                    with dcg.Plot(C, width=-1, height=-1):
                        with dcg.DrawInPlot(C):
                            import imageio
                            image = imageio.imread("lenapashm.png")
                            texture = dcg.Texture(C, image)
                            dcg.DrawImage(C,
                                          texture=texture,
                                          pmin=(0, 1),
                                          pmax=(1, 0))

            with SlideSection(C):
                slideshow.readonly_code(code_demo)

        with TwoColumnSlide(C, title="Edge Detection"):
            with SlideSection(C):
                slideshow.colored_text("OpenCV Integration:")
                code_editor = InteractiveCode(C, initial_code="""
# Edge detection with OpenCV
import numpy as np
import cv2
import imageio

# Load and convert image to grayscale
image = imageio.imread("lenapashm.png")
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Apply Canny edge detection
edges = cv2.Canny(gray, 100, 200)

# Convert back to RGB for display
edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

return edges_rgb""",
                                           width=-1e-3, height=-1e-3)
            with SlideSection(C):
                with CenteredContent(C, vertical_only=True):
                    code_editor.display_result()


        # Update Image Processing Example with histograms
        with TwoColumnSlide(C, title="Image Histograms"):
            with SlideSection(C):
                slideshow.colored_text("Color Channel Analysis:")
                code_editor = InteractiveCode(C, initial_code="""
# Plot RGB channel histograms
import numpy as np
import imageio

# Load image
image = imageio.imread("lenapashm.png")

# Calculate histograms for each channel
bins = np.arange(256)
r_hist = np.histogram(image[:,:,0], bins=bins)[0]
g_hist = np.histogram(image[:,:,1], bins=bins)[0]
b_hist = np.histogram(image[:,:,2], bins=bins)[0]

# Return data for plotting
x = bins[:-1]  # Bin centers
return x, (r_hist, g_hist, b_hist)""",
                                           width=-1e-3, height=-1e-3)
            with SlideSection(C):
                with CenteredContent(C, vertical_only=True):
                    code_editor.display_result()

        # Demonstrating creating an interactable drawing item in a plot
        with Slide(C, title="Example of interactable item"):
            with dcg.Plot(C, width=400, height=400, no_newline=True):
                with dcg.DrawInPlot(C):
                    with dcg.DrawInvisibleButton(C,
                                p1=(0.25, 0.25),
                                p2=(0.75, 0.75)) as btn:
                        dcg.DrawTriangle(C,
                            p1=(0.3, 0.),
                            p2=(1., 0.3),
                            p3=(0.5, 1.),
                            color=(255, 0, 0))
                    def make_temporary_tooltip():
                        with dcg.utils.TemporaryTooltip(C, parent=slideshow):
                            dcg.Text(C, value="Click me!")
                    def change_color(triangle):
                        # Set random color
                        r = np.random.randint(0, 256)
                        g = np.random.randint(0, 256)
                        b = np.random.randint(0, 256)
                        triangle.color = (r, g, b)
                    def make_configuration_window(sender, target):
                        triangle = target.children[0]
                        with dcg.Window(C, label="Change Color",
                                        parent=C.viewport,
                                        popup=True,
                                        autosize=True) as popup:
                            dcg.Button(C, label="Randomize Color",
                                       callback=lambda: change_color(triangle))
                    btn.handlers += [
                        dcg.GotHoverHandler(C, callback=make_temporary_tooltip),
                        dcg.ClickedHandler(C, callback=make_configuration_window)
                    ]
            Gif(C, "simpson.gif", width=400, height=400)

        with Slide(C, title="Interactive maps ?"):
            with dcg.Plot(C, width=-1, height=-1) as map_plot:
                map_plot.Y1.invert = True
                # Remove plot visuals
                map_plot.X1.no_gridlines = True
                map_plot.Y1.no_gridlines = True
                map_plot.X1.no_tick_marks = True
                map_plot.Y1.no_tick_marks = True
                map_plot.X1.no_tick_labels = True
                map_plot.Y1.no_tick_labels = True
                map_plot.no_legend = True
                map_plot.no_mouse_pos = True
                map_plot.no_menus = True
                map_plot.no_title = True
                map_plot.no_frame = True
                map_plot.equal_aspects = True
                map_plot.theme = dcg.ThemeStyleImPlot(C, PlotBorderSize=0)
                """
                DrawMap(C)
                # initialize around paris
                map_plot.X1.min = 2.1
                map_plot.X1.max = 2.6
                map_plot.Y1.min = 48.8
                map_plot.Y1.max = 48.9
                """
                with dcg.DrawInPlot(C):
                    draw_svg = DrawSVG(C, svg_path="Paris_Metro_map.svg")
                    draw_svg_complete = DrawSVG(C, svg_path="Paris_department_land_cover_location_map.svg", show=False)
                map_plot.X1.fit()
                def switch_map():
                    if draw_svg_complete.show:
                        draw_svg.show = True
                        draw_svg_complete.show = False
                    else:
                        draw_svg.show = False
                        draw_svg_complete.show = True
            map_plot.handlers += [
                dcg.ClickedHandler(C, button=dcg.MouseButton.RIGHT,
                                    callback=switch_map)
            ]


        # Resources
        with Slide(C, title="Resources"):
            with CenteredContent(C):
                slideshow.colored_text("Learn More:")
                dcg.Text(C, value="Documentation: github.com/DearCyGui/DearCyGui/docs")
                dcg.Text(C, value="GitHub Repository: github.com/DearCyGui/DearCyGui")
                dcg.Text(C, value="Demos: github.com/DearCyGui/Demos")
                slideshow.separator()
                dcg.Text(C, value="Thank you!")

    # Initialize viewport and run
    C.viewport.initialize(title="DearCyGui for Scientific Applications", vsync=True, wait_for_input=True)
    
    slideshow.start()
    #export_slideshow(_get_unique_filename("slides", "pdf"), slideshow)
    
    # Main loop
    while C.running:
        C.viewport.render_frame(can_skip_presenting=True)

if __name__ == "__main__":
    slideshow_dearcygui()
