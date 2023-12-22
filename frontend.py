import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import csv

from api import Result
from custom_widget import *


COLOR_BACKGROUND = "#3a615c"
COLOR_FILL_LGT = "#82b0a2"
COLOR_FILL_MED = "#e4eaf5"
COLOR_TXT_DARK = "#0c0e12"

FONT_1 = ("Calibri", 15, "bold")


class FeatureConfig(tk.Frame):

    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        # Widget initialization (TO DO: needs to be cleaner!)
        self.entry_path = tk.Entry(self, width=50, bg=COLOR_FILL_MED)
        self.browse_button = ttk.Button(self, text="Browse", 
                                        command=self.browse_file)

        self.columns = tk.Listbox(self, selectmode=tk.MULTIPLE, 
                                  height=15, bg=COLOR_FILL_LGT, 
                                  selectbackground="#254a3a")
        self.label_columns = ttk.Label(self, text="Columns")
        self.button_remove = CustomButton(self, self.parent.manager, 
                                          text="<--", state="disabled", 
                                          command=lambda: self.move_selected("remove"))

        self.target = tk.Listbox(self, selectmode=tk.MULTIPLE, 
                                 height=1, bg=COLOR_FILL_LGT, 
                                 selectbackground="#254a3a")
        self.label_target = ttk.Label(self, text="Target", 
                                      style="TLabel")
        self.button_target = CustomButton(self, self.parent.manager, 
                                          text="-->", state="disabled",
                                          command=lambda: self.move_selected("target"))

        self.features = tk.Listbox(self, selectmode=tk.MULTIPLE, 
                                   bg=COLOR_FILL_LGT, 
                                   selectbackground="#254a3a")
        self.label_features = ttk.Label(self, text="Features") 
        self.button_features = CustomButton(self, self.parent.manager, 
                                            text="-->", state="disabled",
                                            command=lambda: self.move_selected("features"))

        # Widget placement
        self.entry_path.grid(row=0, column=0, padx=10, pady=10, 
                             columnspan=3, sticky="W")
        self.browse_button.grid(row=0, column=3, padx=10, 
                                pady=10, sticky="W")

        self.columns.grid(row=2, column=0, padx=10, pady=0, 
                          rowspan=4, sticky="N")
        self.label_columns.grid(row=1, column=0, padx=10, 
                                pady=0, sticky="S")
        self.button_remove.grid(row=3, column=1, padx=10, 
                                pady=10)

        self.target.grid(row=2, column=3, padx=10, pady=0, 
                         sticky="N")
        self.label_target.grid(row=1, column=3, padx=10, 
                               pady=0, sticky="S")
        self.button_target.grid(row=2, column=2, padx=10, 
                                pady=0, sticky="N")

        self.features.grid(row=4, column=3, padx=10, pady=0, sticky="N")
        self.label_features.grid(row=3, column=3, padx=10, pady=0, sticky="S")
        self.button_features.grid(row=4, column=2, padx=10, pady=10)


    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        self.entry_path.delete(0, tk.END)
        self.entry_path.insert(0, file_path)
        self.get_csv(file_path)

    def get_csv(self, file_path):

        if file_path:
            file_name = file_path.split("/")[-1]
            with open(file_path, "r") as file:
                reader = csv.reader(file)
                self.validate_csv(reader, file)
            self.show_columns(self.header)
            self.parent.manager.enable_all()
            self.parent.terminal.write(f"======| {file_name} uploaded successfully |======")
        #     except FileNotFoundError:
        #         print(f"CSV file not found: {file_path}")
        #     except ValueError as e:
        #         self.parent.terminal.write(e)
        #     # except Exception as e:
        #     #     self.parent.terminal.write(f"Unknown error: {file_name}")
        # else:
        #     self.parent.terminal.write("No CSV chosen.")

    def validate_csv(self, reader, data):
        try:
            # Load the CSV file using np.genfromtxt
            self.header = ["_".join(i.split()) for i in next(reader)]
            self.raw_data = np.genfromtxt(data, dtype=None, delimiter=",", names=self.header, 
                                          encoding=None)

            # Check for inconsistent column lengths
            # expected_num_columns = len(data.dtype.names)
            # actual_num_columns = np.max([len(row) for row in data])
            # if expected_num_columns != actual_num_columns:
            #     raise ValueError(f"Error: Inconsistent column lengths. Expected {expected_num_columns} columns, but found {actual_num_columns} columns.")

        except np.lib._iotools.ConversionWarning as e:
            print(f"Error: {e}")
            return None

    def show_columns(self, columns):
        # Clear existing listboxes
        self.columns.delete(0, tk.END)
        self.features.delete(0, tk.END)
        self.target.delete(0, tk.END)

        # Display column names in listbox1
        for col_name in columns:
            self.columns.insert(tk.END, col_name)

    def move_selected(self, direction):

        if direction == "target":
            selected_items = self.columns.curselection()
            for index in reversed(selected_items):
                self.target.insert(tk.END, self.columns.get(index))
                self.columns.delete(index)

        elif direction == "features":
            selected_items = self.columns.curselection()
            for index in reversed(selected_items):
                self.features.insert(tk.END, self.columns.get(index))
                self.columns.delete(index)

        elif direction == "remove":
            selected_items_target = self.target.curselection()
            for index in reversed(selected_items_target):
                self.columns.insert(tk.END, self.target.get(index))
                self.target.delete(index)

            selected_items_features = self.features.curselection()
            for index in reversed(selected_items_features):
                self.columns.insert(tk.END, self.features.get(index))
                self.features.delete(index)

class ModelConfig(tk.Frame):
    
    def __init__(self, parent, **kwargs):
        tk.Frame.__init__(self, parent, **kwargs)
        self.grid_rowconfigure(1, weight=1)
        self.parent = parent

        # self.model_selection = ModelSelection(self, self.manager, text="Model type:", style="TLabelframe")
        self.model_type = ModelType(self, self.parent.manager, text="Model type:")
        self.options = Options(self, self.parent.manager, text="Options:")
        self.button_execute = CustomButton(self, self.parent.manager, text="Execute", command=self.start_execute, state="disabled")
        
        self.model_type.grid(row=0, column=0, padx=10, pady=10)
        self.options.grid(row=1, column=0, padx=10, pady=10)
        self.button_execute.grid(row=2, column=0, padx=10, pady=10, sticky="S")

    def start_execute(self):

        feature_listbox = self.parent.feature_config.features
        feature_content = [feature_listbox.get(index) for index in range(feature_listbox.size())]
        target_listbox = self.parent.feature_config.target
        target_content = [target_listbox.get(index) for index in range(target_listbox.size())]
        model_type = self.parent.model_config.model_type.model_type_final

        try:
            self.validate_feature_config(feature_content, target_content)
            self.validate_model_config(model_type)

            raw_data = self.parent.feature_config.raw_data

            feature_used = [raw_data[i] for i in feature_content]
            target_used = raw_data[target_content[0]]

            result_window = ResultWindow(self.parent, feature_used, target_used, bg=COLOR_BACKGROUND)

        except ValueError as e:  # Catch specific exceptions for better error handling
           self.parent.terminal.write(f"config error: {str(e)}")

    def validate_feature_config(self, feature, target):
       if len(feature) == 0:
           raise ValueError("no feature chosen")
       if len(target) == 0:
           raise ValueError("no target chosen")
       if len(target) >= 2:
           raise ValueError("target more than one")

    def validate_model_config(self, model_type):
       if len(model_type.get()) == 0:
           raise ValueError("no model type chosen")



class Options(ttk.LabelFrame):

    def __init__(self, parent, manager, **kwargs):
        ttk.LabelFrame.__init__(self, parent, **kwargs)
        self.manager = manager

        self.run_pca = tk.BooleanVar()
        self.save = tk.BooleanVar()

        self.check_opt_1 = CustomCheck(self, self.manager, text="Run PCA", 
                                       state="disabled", variable=self.run_pca,)
        self.check_opt_2 = CustomCheck(self, self.manager, text="Save result", 
                                       state="disabled", variable=self.save)

        self.check_opt_1.grid(padx=10, sticky="NW")
        self.check_opt_2.grid(padx=10, sticky="NW")

class ModelType(ttk.LabelFrame):

    def __init__(self, parent, manager, **kwargs):
        ttk.LabelFrame.__init__(self, parent, **kwargs)
        self.manager = manager

        self.model_type_final = tk.StringVar()

        self.radio_mod_type_1 = CustomRadio(self, self.manager, text="Regression", value="Regression", 
                                            state="disabled", variable=self.model_type_final)
        self.radio_mod_type_2 = CustomRadio(self, self.manager, text="Classification", value="Classification",
                                            state="disabled", variable=self.model_type_final)

        self.radio_mod_type_1.grid(padx=10, sticky="NW")
        self.radio_mod_type_2.grid(padx=10, sticky="NW")

class ResultWindow(tk.Toplevel):

    def __init__(self, parent, feature, target, **kwargs):

        tk.Toplevel.__init__(self, parent, **kwargs)
        self.iconbitmap("SP.ico")
        self.parent = parent
        result = Result(feature, target, None)
        self.models = result.get_models()
        model_names = list(self.models.keys())

        results = []

        for i, j in self.models.items():

            # Update to break from the single thread lock
            self.parent.terminal.write(f"running {i}...")
            self.parent.update_idletasks()

            # Running the main calculation
            results.append(j.fit().evaluate())

        parent.terminal.write("======| Execution successful |======")
        column_list = ["algorithm"] + list(results[0].keys())
        tree = SortableTreeview(self, columns=column_list, show='headings')
        tree.grid(padx=10, pady=10)

        # Set the headings
        for col in column_list:
            tree.heading(col, text=col)
            if col=="algorithm":
                tree.column(col, width=170)  # Adjust the width as needed
            else:
                tree.column(col, width=100, anchor="center")

        # Populate the treeview with data from the list of dictionaries
        for i, data_dict in enumerate(results):
            tree.insert('', 'end', values=tuple([model_names[i]] + list(data_dict.values())))

        
class Terminal(tk.Frame):

    def __init__(self, parent, **kwargs):
        tk.Frame.__init__(self, parent, **kwargs)

        self.text_widget = tk.Text(self, wrap=tk.WORD, state=tk.DISABLED, 
                                   bg=COLOR_FILL_MED, height=5, width=55,
                                   font=("Courier New", 9))
        self.label_terminal = tk.Label(self, text="Status", bg=COLOR_BACKGROUND, font=("arial", 10, "bold"))

        self.label_terminal.pack(side=tk.TOP)
        self.text_widget.pack(fill=tk.BOTH, padx=10, pady=3)

    def write(self, output):
        # Enable the text widget to modify its content
        self.text_widget.config(state=tk.NORMAL)

        # Insert the output at the end of the text widget
        self.text_widget.insert(tk.END, output + "\n")

        # Disable the text widget again to make it read-only
        self.text_widget.config(state=tk.DISABLED)

        # Auto-scroll to the end
        self.text_widget.yview(tk.END)
        


class MainApplication(tk.Frame):

    def __init__(self, parent, *args, **kwargs):

        tk.Frame.__init__(self, parent, *args, **kwargs)  # packing the Frame

        # Styling the widgets
        style = ttk.Style(self)
        style.configure("TLabel", font=("Calibri", 15, "bold"), 
                        background=COLOR_BACKGROUND, fg=COLOR_TXT_DARK)
        style.configure("TRadiobutton", background="#014647", foreground="white", font=("Arial", 10))
        style.configure("TCheckbutton", background="#014647", foreground="white", font=("Arial", 10))
        style.configure("TLabelframe", background="#014647")
        style.configure("TLabelframe.Label", background="#014647", foreground="white", font=("Arial", 10))
        style.configure("TButton", font=("calibri", 10, "bold"), width=6)
        style.map("TButton",
                  background=[("active", "#254a3a"), ("disabled", "#888888")],
                  foreground=[("active", "black"), ("disabled", "gray")])
        style.configure("Treeview", font=("Calibri", 10), fg=COLOR_TXT_DARK)
        style.configure("Treeview.Heading", font=("Calibri", 12, "bold"), fg=COLOR_BACKGROUND, bg="black")


        # Setting the main
        self.parent = parent
        self.manager = WidgetManager()
        self.terminal = Terminal(self, bg=COLOR_BACKGROUND)
        self.feature_config = FeatureConfig(self, bg=COLOR_BACKGROUND)
        self.model_config = ModelConfig(self, bg="#014647")
        

        self.feature_config.grid(row=0, column=0)
        self.model_config.grid(row=0, column=1, rowspan=2, sticky=tk.NSEW)
        self.terminal.grid(row=1, column=0, columnspan=1, sticky=tk.NSEW)


if __name__ == "__main__":
    root = tk.Tk()
    app = MainApplication(root)
    app.grid()
    root.title("SuperPlug")
    root.iconbitmap("SP.ico")
    root.configure(background=COLOR_BACKGROUND)
    root.resizable(False, False)
    root.mainloop()