import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QSlider, QLabel, QPushButton, QGroupBox,
                             QSpinBox, QDoubleSpinBox, QMessageBox, QComboBox, QStyledItemDelegate)
from PyQt5.QtCore import Qt, QPointF, QSize
from PyQt5.QtGui import QPainter, QPen, QColor, QPixmap, QLinearGradient, QIcon
import pyqtgraph as pg
from scipy.optimize import curve_fit
import nibabel as nib
from utils import calculate_t2_map

class ColormapDelegate(QStyledItemDelegate):
    """Custom delegate to show colormap gradients in combo box"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.colormap_colors = {
            'viridis': [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)],
            'plasma': [(13, 8, 135), (126, 3, 168), (204, 71, 120), (248, 149, 64), (240, 249, 33)],
            'inferno': [(0, 0, 4), (87, 16, 110), (188, 55, 84), (249, 142, 9), (252, 255, 164)],
            'magma': [(0, 0, 4), (80, 18, 123), (182, 54, 121), (251, 136, 97), (252, 253, 191)],
            'jet': [(0, 0, 143), (0, 0, 255), (0, 255, 255), (255, 255, 0), (255, 0, 0)],
            'hot': [(0, 0, 0), (128, 0, 0), (255, 128, 0), (255, 255, 128), (255, 255, 255)],
            'cool': [(0, 255, 255), (64, 191, 255), (128, 128, 255), (191, 64, 255), (255, 0, 255)],
            'rainbow': [(127, 0, 255), (0, 0, 255), (0, 255, 0), (255, 255, 0), (255, 0, 0)],
            'turbo': [(48, 18, 59), (62, 96, 213), (33, 179, 142), (182, 232, 35), (122, 4, 2)],
            'hsv': [(255, 0, 0), (255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 255)]
        }
    
    def paint(self, painter, option, index):
        """Paint the combo box item with gradient"""
        painter.save()
        
        # Draw background
        if option.state & 0x00000001:  # Selected
            painter.fillRect(option.rect, QColor(100, 100, 180))
        else:
            painter.fillRect(option.rect, option.palette.base())
        
        # Get colormap name
        colormap_name = index.data()
        
        if colormap_name in self.colormap_colors:
            # Draw gradient
            gradient_rect = option.rect.adjusted(5, 5, -100, -5)
            gradient = QLinearGradient(gradient_rect.topLeft(), gradient_rect.topRight())
            
            colors = self.colormap_colors[colormap_name]
            for i, color in enumerate(colors):
                position = i / (len(colors) - 1)
                gradient.setColorAt(position, QColor(*color))
            
            painter.setBrush(gradient)
            painter.setPen(Qt.NoPen)
            painter.drawRect(gradient_rect)
            
            # Draw border around gradient
            painter.setPen(QPen(QColor(128, 128, 128), 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(gradient_rect)
        
        # Draw text
        text_rect = option.rect.adjusted(option.rect.width() - 95, 0, -5, 0)
        painter.setPen(option.palette.text().color())
        painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignVCenter, colormap_name)
        
        painter.restore()
    
    def sizeHint(self, option, index):
        """Return size hint for items"""
        return QSize(option.rect.width(), 30)
    

class T2MappingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("T2 Mapping Application")
        self.setGeometry(100, 100, 1600, 900)
        
        # Initialize data parameters
        self.img_size = 128
        self.n_echoes = 10
        self.n_slices = 24
        self.te_values = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])  # TE values in ms
        
        # State variables
        self.current_slice = 0
        self.current_te = 0
        self.noise_level = 0.0
        self.t2_map = None
        self.s0_map = None
        self.noisy_data = None
        self.noisy_t2_map = None
        self.noisy_s0_map = None
        self.selected_pixel = None
        self.roi_points = []
        self.roi_active = False
        self.current_t2_color_map = 'viridis'

        self.load_image()
        
        self.init_ui()
        self.update_display()

    def load_image(self, file_path='prostate_010.nii.gz'):
        img = img = nib.load(file_path)
        data = img.get_fdata()
        self.te_values = np.linspace(13.2, 145.2, 10)
        self.t2_map, self.s0_map = calculate_t2_map(data[:, :, self.current_slice, :], self.te_values / 1000)
        self.original_data = data.copy()
        self.data = data.copy()
        self.n_slices = data.shape[2]
        self.n_echoes = data.shape[3]

        
    # def generate_synthetic_data(self):
    #     """Generate synthetic T2-weighted MRI data"""
    #     # Create ground truth T2 map with different regions
    #     x, y = np.meshgrid(np.linspace(-1, 1, self.img_size), 
    #                       np.linspace(-1, 1, self.img_size))
    #     r = np.sqrt(x**2 + y**2)
        
    #     # Create regions with different T2 values
    #     t2_true = np.zeros((self.img_size, self.img_size))
    #     t2_true[r < 0.3] = 80  # Center region
    #     t2_true[(r >= 0.3) & (r < 0.6)] = 50  # Middle ring
    #     t2_true[(r >= 0.6) & (r < 0.8)] = 30  # Outer ring
    #     t2_true[r >= 0.8] = 10  # Background
        
    #     # Generate signal intensities based on T2 decay: S = S0 * exp(-TE/T2)
    #     self.data = np.zeros((self.n_echoes, self.img_size, self.img_size))
    #     s0 = 1000  # Proton density
        
    #     for i, te in enumerate(self.te_values):
    #         self.data[i] = s0 * np.exp(-te / t2_true)
        
    #     self.original_data = self.data.copy()
    #     self.true_t2 = t2_true
        
    def init_ui(self):
        """Initialize the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - Controls
        left_panel = QVBoxLayout()
        
        # TE Slider
        te_group = QGroupBox("Echo Time (TE)")
        te_layout = QVBoxLayout()
        self.te_slider = QSlider(Qt.Horizontal)
        self.te_slider.setMinimum(0)
        self.te_slider.setMaximum(self.n_echoes - 1)
        self.te_slider.setValue(0)
        self.te_slider.valueChanged.connect(self.on_te_changed)
        self.te_label = QLabel(f"TE: {self.te_values[0]} ms")
        te_layout.addWidget(self.te_label)
        te_layout.addWidget(self.te_slider)
        te_group.setLayout(te_layout)
        left_panel.addWidget(te_group)

        # Slice Slider
        slice_group = QGroupBox("Slice Number")
        slice_layout = QVBoxLayout()
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(self.n_slices - 1)
        self.slice_slider.setValue(0)
        self.slice_slider.valueChanged.connect(self.on_slice_changed)
        slice_layout.addWidget(self.slice_slider)
        slice_group.setLayout(slice_layout)
        left_panel.addWidget(slice_group)
        
        # Noise Control
        noise_group = QGroupBox("Noise Control")
        noise_layout = QVBoxLayout()
        noise_hlayout = QHBoxLayout()
        noise_hlayout.addWidget(QLabel("Noise Level (%):"))
        self.noise_spinbox = QDoubleSpinBox()
        self.noise_spinbox.setRange(0, 50)
        self.noise_spinbox.setValue(0)
        self.noise_spinbox.setSingleStep(1)
        noise_hlayout.addWidget(self.noise_spinbox)
        noise_layout.addLayout(noise_hlayout)
        self.add_noise_btn = QPushButton("Add Noise")
        self.add_noise_btn.clicked.connect(self.add_noise)
        noise_layout.addWidget(self.add_noise_btn)
        noise_group.setLayout(noise_layout)
        left_panel.addWidget(noise_group)
        
        # T2 Mapping Control
        t2_group = QGroupBox("T2 Mapping")
        t2_layout = QVBoxLayout()
        self.calc_t2_btn = QPushButton("Calculate T2 Map")
        self.calc_t2_btn.clicked.connect(self.calculate_t2_map)
        t2_layout.addWidget(self.calc_t2_btn)
        
        # Color scheme selector
        colormap_layout = QHBoxLayout()
        colormap_layout.addWidget(QLabel("Colormap:"))
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(['viridis', 'plasma', 'inferno', 'magma', 'jet', 
                                       'hot', 'cool', 'rainbow', 'turbo', 'hsv'])
        
        # Set custom delegate to show gradients
        self.colormap_delegate = ColormapDelegate(self.colormap_combo)
        self.colormap_combo.setItemDelegate(self.colormap_delegate)
        self.colormap_combo.setIconSize(QSize(100, 20))
        
        self.colormap_combo.currentTextChanged.connect(self.on_colormap_changed)
        colormap_layout.addWidget(self.colormap_combo)
        t2_layout.addLayout(colormap_layout)
        
        t2_group.setLayout(t2_layout)
        left_panel.addWidget(t2_group)
        
        # ROI Control
        roi_group = QGroupBox("ROI Analysis")
        roi_layout = QVBoxLayout()
        self.roi_btn = QPushButton("Draw ROI")
        self.roi_btn.setCheckable(True)
        self.roi_btn.clicked.connect(self.toggle_roi_mode)
        roi_layout.addWidget(self.roi_btn)
        self.clear_roi_btn = QPushButton("Clear ROI")
        self.clear_roi_btn.clicked.connect(self.clear_roi)
        roi_layout.addWidget(self.clear_roi_btn)
        self.roi_stats_label = QLabel("ROI Stats:\nNo ROI selected")
        roi_layout.addWidget(self.roi_stats_label)
        roi_group.setLayout(roi_layout)
        left_panel.addWidget(roi_group)
        
        left_panel.addStretch()
        
        # Middle panel - Images
        middle_panel = QVBoxLayout()
        
        # Original/Noisy Image
        self.img_widget = pg.ImageView()
        self.img_widget.ui.roiBtn.hide()
        self.img_widget.ui.menuBtn.hide()
        self.img_widget.getView().scene().sigMouseClicked.connect(self.on_image_clicked)
        middle_panel.addWidget(QLabel("Original/Noisy Image"))
        middle_panel.addWidget(self.img_widget)
        
        # T2 Map
        self.t2_widget = pg.ImageView()
        self.t2_widget.ui.roiBtn.hide()
        self.t2_widget.ui.menuBtn.hide()
        middle_panel.addWidget(QLabel("T2 Map"))
        middle_panel.addWidget(self.t2_widget)
        
        # Right panel - Plots and difference map
        right_panel = QVBoxLayout()
        
        # Signal decay plot
        self.signal_plot = pg.PlotWidget(title="Signal Decay at Selected Pixel")
        self.signal_plot.setLabel('left', 'Signal Intensity')
        self.signal_plot.setLabel('bottom', 'TE (ms)')
        self.signal_plot.addLegend()
        right_panel.addWidget(self.signal_plot)
        
        # Difference map
        self.diff_widget = pg.ImageView()
        self.diff_widget.ui.roiBtn.hide()
        self.diff_widget.ui.menuBtn.hide()
        right_panel.addWidget(QLabel("Difference Map (Original - Noisy)"))
        right_panel.addWidget(self.diff_widget)
        
        # Add panels to main layout
        main_layout.addLayout(left_panel, 1)
        main_layout.addLayout(middle_panel, 2)
        main_layout.addLayout(right_panel, 2)

    def on_colormap_changed(self, colormap_name):
        """Handle colormap selection change"""
        self.current_t2_color_map = colormap_name
        self.update_display()

    def on_slice_changed(self, value):
        self.current_slice = value
        self.update_display()
        
    def on_te_changed(self, value):
        """Handle TE slider change"""
        self.current_te = value
        self.te_label.setText(f"TE: {self.te_values[value]:.2f} ms")
        self.update_display()
        
    def add_noise(self):
        """Add Gaussian noise to the data"""
        self.noise_level = self.noise_spinbox.value()
        if self.noise_level == 0:
            self.noisy_data = None
            self.noisy_t2_map = None
        else:
            # Add Gaussian noise as percentage of signal
            noise_std = self.noise_level / 100.0
            self.noisy_data = self.original_data.copy()
            for i in range(self.n_echoes):
                noise = np.random.normal(0, noise_std * np.mean(self.original_data[i]), 
                                        (self.img_size, self.img_size))
                self.noisy_data[i] = self.original_data[i] + noise
                self.noisy_data[i] = np.maximum(self.noisy_data[i], 0)  # Ensure non-negative
        
        self.update_display()
        if self.selected_pixel:
            self.plot_signal_decay()
            
    def calculate_t2_map(self):
        """Calculate T2 map using exponential fitting"""
        # Use noisy data if available, otherwise original
        data_to_fit = self.noisy_data if self.noisy_data is not None else self.original_data
        
        t2_map = np.zeros((self.img_size, self.img_size))
        
        # Fit T2 for each pixel
        for i in range(self.img_size):
            for j in range(self.img_size):
                signal = data_to_fit[:, i, j]
                try:
                    # Fit exponential decay: S = S0 * exp(-TE/T2)
                    popt, _ = curve_fit(lambda te, s0, t2: s0 * np.exp(-te / t2),
                                       self.te_values, signal,
                                       p0=[signal[0], 50],
                                       bounds=([0, 1], [np.inf, 200]),
                                       maxfev=1000)
                    t2_map[i, j] = popt[1]
                except:
                    t2_map[i, j] = 0
        
        if self.noisy_data is not None:
            self.noisy_t2_map = t2_map
        else:
            self.t2_map = t2_map
            
        self.update_display()
        QMessageBox.information(self, "T2 Mapping", "T2 map calculation complete!")
        
    def on_image_clicked(self, event):
        """Handle mouse click on image"""
        pos = event.scenePos()
        if self.img_widget.getImageItem().sceneBoundingRect().contains(pos):
            mouse_point = self.img_widget.getView().mapSceneToView(pos)
            x, y = int(mouse_point.x()), int(mouse_point.y())
            
            if 0 <= x < self.img_size and 0 <= y < self.img_size:
                if self.roi_active:
                    self.roi_points.append((y, x))
                    self.update_display()
                    if len(self.roi_points) >= 3:
                        self.calculate_roi_stats()
                else:
                    self.selected_pixel = (y, x)
                    self.plot_signal_decay()
                    
    def plot_signal_decay(self):
        """Plot signal decay curve for selected pixel"""
        if self.selected_pixel is None:
            return
            
        y, x = self.selected_pixel
        self.signal_plot.clear()
        
        # Plot original signal
        original_signal = self.original_data[x, y, self.current_slice, :]
        self.signal_plot.plot(self.te_values, original_signal, 
                             pen=pg.mkPen('b', width=2), 
                             symbol='o', symbolBrush='b',
                             name='Original Signal')
        
        # Plot noisy signal if available
        if self.noisy_data is not None:
            noisy_signal = self.noisy_data[x, y, self.current_slice, :]
            self.signal_plot.plot(self.te_values, noisy_signal,
                                 pen=pg.mkPen('r', width=2),
                                 symbol='s', symbolBrush='r',
                                 name='Noisy Signal')
        
        # Plot fitted curves if T2 maps exist
        te_fine = np.linspace(self.te_values[0], self.te_values[-1], 100)
        
        if self.t2_map is not None:
            t2_orig = self.t2_map[x, y]
            s0_orig = self.s0_map[x, y]
            fitted_orig = s0_orig * np.exp(-te_fine / (t2_orig * 1000))
            self.signal_plot.plot(te_fine, fitted_orig,
                                 pen=pg.mkPen('c', width=2, style=Qt.DashLine),
                                 name=f'Fit (T2={t2_orig:.3f}s)')
        
        if self.noisy_t2_map is not None and self.noisy_data is not None:
            t2_noisy = self.noisy_t2_map[x, y]
            s0_noisy = self.noisy_s0_map[x, y]
            fitted_noisy = s0_noisy * np.exp(-te_fine / t2_noisy)
            self.signal_plot.plot(te_fine, fitted_noisy,
                                 pen=pg.mkPen('m', width=2, style=Qt.DashLine),
                                 name=f'Noisy Fit (T2={t2_noisy:.1f}ms)')
        
    def toggle_roi_mode(self):
        """Toggle ROI drawing mode"""
        self.roi_active = self.roi_btn.isChecked()
        if self.roi_active:
            self.roi_points = []
            self.roi_btn.setText("Drawing ROI... (Click to add points)")
        else:
            self.roi_btn.setText("Draw ROI")
            
    def clear_roi(self):
        """Clear ROI selection"""
        self.roi_points = []
        self.roi_active = False
        self.roi_btn.setChecked(False)
        self.roi_btn.setText("Draw ROI")
        self.roi_stats_label.setText("ROI Stats:\nNo ROI selected")
        self.update_display()
        
    def calculate_roi_stats(self):
        """Calculate statistics within ROI"""
        if len(self.roi_points) < 3:
            return
            
        # Create mask from ROI points
        from matplotlib.path import Path
        roi_path = Path([(x, y) for y, x in self.roi_points])
        
        y_grid, x_grid = np.meshgrid(np.arange(self.img_size), np.arange(self.img_size))
        points = np.vstack((x_grid.flatten(), y_grid.flatten())).T
        mask = roi_path.contains_points(points).reshape(self.img_size, self.img_size)
        
        stats_text = "ROI Stats:\n"
        
        # Calculate T2 statistics for original data
        if self.t2_map is not None:
            roi_t2_orig = self.t2_map[mask]
            stats_text += f"Original T2: {np.mean(roi_t2_orig):.2f} ± {np.std(roi_t2_orig):.2f} ms\n"
        
        # Calculate T2 statistics for noisy data
        if self.noisy_t2_map is not None:
            roi_t2_noisy = self.noisy_t2_map[mask]
            stats_text += f"Noisy T2: {np.mean(roi_t2_noisy):.2f} ± {np.std(roi_t2_noisy):.2f} ms\n"
            
            if self.t2_map is not None:
                diff = np.mean(roi_t2_orig) - np.mean(roi_t2_noisy)
                stats_text += f"Mean Difference: {diff:.2f} ms"
        
        self.roi_stats_label.setText(stats_text)
        
    def update_display(self):
        """Update all image displays"""
        # Display current image
        current_data = self.noisy_data if self.noisy_data is not None else self.original_data
        self.img_widget.setImage(current_data[:, :, self.current_slice, self.current_te], autoRange=False, autoLevels=False)
        
        # Display T2 map
        t2_to_show = self.noisy_t2_map if self.noisy_t2_map is not None else self.t2_map
        if t2_to_show is not None:
            # Apply colormap - try matplotlib first, fall back to pyqtgraph built-in
            try:
                colormap = pg.colormap.get(self.current_colormap, source='matplotlib')
            except:
                # Fall back to pyqtgraph colormaps if matplotlib not available
                colormap = pg.colormap.get(self.current_t2_color_map)
            
            if colormap is not None:
                self.t2_widget.setColorMap(colormap)

            self.t2_widget.setImage(t2_to_show, autoRange=False, autoLevels=False)
        
        # Display difference map
        if self.noisy_data is not None:
            diff_map = self.original_data[:, :, self.current_slice, self.current_te] - self.noisy_data[self.current_slice, :, :, self.current_te]
            self.diff_widget.setImage(diff_map, autoRange=False, autoLevels=False)
        
        # Draw ROI if points exist
        if self.roi_points:
            self.draw_roi_overlay()
            
    def draw_roi_overlay(self):
        """Draw ROI overlay on image"""
        if not self.roi_points:
            return
        
        # This is a simplified visualization - for production you'd want to use
        # pyqtgraph's ROI tools or custom graphics items
        pass  # ROI visualization handled by click feedback

def main():
    app = QApplication(sys.argv)
    window = T2MappingApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
