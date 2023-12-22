from tkinter import ttk

class SortableTreeview(ttk.Treeview):
    def __init__(self, master=None, columns=None, **kwargs):
        ttk.Treeview.__init__(self, master, columns=columns, **kwargs)
        self.heading_clicked = {col: False for col in columns}
        self.current_sorted_column = None  # Track the currently sorted column

        for col in columns:
            self.heading(col, text=col, command=lambda c=col: self.sort_column(c, False))

    def sort_column(self, col, reverse):
        items = [(self.set(item, col), item) for item in self.get_children('')]

        # Determine the data type for sorting
        data_type = str
        for item in items:
            try:
                float(self.set(item[1], col))
                data_type = float
            except ValueError:
                pass

        items.sort(key=lambda x: data_type(x[0]), reverse=reverse)

        for index, (val, item) in enumerate(items):
            self.move(item, '', index)

        # Update visual cues for the sorted column
        self.update_sort_cues(col, not reverse)

        # Toggle the heading_clicked dictionary to alternate between ascending and descending
        self.heading_clicked[col] = not reverse
        self.heading(col, command=lambda: self.sort_column(col, not reverse))

    def update_sort_cues(self, col, reverse):
        # Clear sort cues for all columns
        for column in self["columns"]:
            self.heading(column, text=self.heading(column)["text"].rstrip(" ↑↓"))

        # Set arrow icon based on the sorting order
        arrow_icon = " ↓" if reverse else " ↑"
        self.heading(col, text=self.heading(col)["text"] + arrow_icon)

        # Highlight the sorted column
        self.heading(col)
        if self.current_sorted_column and self.current_sorted_column != col:
            self.heading(self.current_sorted_column)

        self.current_sorted_column = col



# Setting custom widget by inheriting from the original

class WidgetManager():

    def __init__(self):
        self.widgets = []

    def add_widget(self, widget):
        self.widgets.append(widget)

    def enable_all(self):
        for widget in self.widgets:
            widget.enable()

    def disable_all(self):
        for widget in self.widgets:
            widget.disable()

class CustomButton(ttk.Button):

    def __init__(self, parent, manager, **kwargs):
        ttk.Button.__init__(self, parent, **kwargs)
        self.manager = manager
        self.manager.add_widget(self)

    def enable(self):
        self['state'] = 'normal'

    def disable(self):
        self['state'] = 'disabled'

class CustomRadio(ttk.Radiobutton):

    def __init__(self, parent, manager, **kwargs):
        ttk.Radiobutton.__init__(self, parent, **kwargs)
        self.manager = manager
        self.manager.add_widget(self)

    def enable(self):
        self['state'] = 'normal'

    def disable(self):
        self['state'] = 'disabled'

class CustomCheck(ttk.Checkbutton):

    def __init__(self, parent, manager, **kwargs):
        ttk.Checkbutton.__init__(self, parent, **kwargs)
        self.manager = manager
        self.manager.add_widget(self)

    def enable(self):
        self['state'] = 'normal'

    def disable(self):
        self['state'] = 'disabled'