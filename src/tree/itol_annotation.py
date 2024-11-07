"""
Utilities to generate iTOL [1] annotations.

[1] https://itol.embl.de
"""
import csv
import os
import textwrap
from typing import Union, List, Optional, Callable

from Bio import SeqIO
import numpy as np
import seaborn as sns


def itol_labels(
    data: List[List[str]],
    output_path: Union[str, bytes, os.PathLike],
) -> None:
    """
    Create iTOL label annotations.

    Example data input: 
    ```
    data = [
        ['GCA_001294575.1', 'Streptococcus epidemicus'],
        ['GCA_000761155.1', 'Halogranum salarium'],
    ]
    ```

    - Format: 
        https://itol.embl.de/help.cgi#labels
    - Template: 
        https://itol.embl.de/help/labels_template.txt
    """
    metadata = textwrap.dedent(f"""\
    LABELS
    SEPARATOR COMMA
    DATA
    """)
    do_write(output_path, metadata, data)


def itol_binary_annotations(
    data: List[List[str]],
    output_path: Union[str, bytes, os.PathLike],
    field_shapes: List[int],
    field_labels: List[str],
    dataset_label: str = 'Dataset',
    dataset_color: str = '#d55e00',
    field_colors: List[str] = None,
    legend_title: str = 'Shapes Legend',
    margin: int = 5,
    width: int = 20,
    border_width: float = 0.0,
    symbol_spacing: int = 10,
    height_factor: float = 1,
) -> None:
    """
    Create iTOL binary annotations.

    Example data input: 
    ```
    data = [
        ['GCA_001294575.1', '1'],
        ['GCA_000761155.1', '-1'],
    ]
    ```

    - Format: 
        https://itol.embl.de/help.cgi#binary
    - Template: 
        https://itol.embl.de/help/dataset_binary_template.txt
    """
    if len(data) == 0:
        raise ValueError('Data is empty')

    n_elements = len(data[0]) - 1
    if n_elements <= 0:
        raise ValueError('Data elements must contain at least 2 elements')
    elif len(field_shapes) != n_elements:
        v = len(field_shapes)
        raise ValueError(f'Expected {n_elements} field shapes, not {v}')
    elif len(field_labels) != n_elements:
        v = len(field_labels)
        raise ValueError(f'Expected {n_elements} field labels, not {v}')

    palette = sns.color_palette('colorblind', n_elements).as_hex()
    if field_colors is None:
        field_colors = [palette[i] for i in range(n_elements)]
    elif len(field_colors) != n_elements:
        v = len(field_colors)
        raise ValueError(f'Expected {n_elements} field colors, not {v}')

    metadata = textwrap.dedent(f"""\
    DATASET_BINARY
    SEPARATOR COMMA
    DATASET_LABEL,{dataset_label}
    COLOR,{dataset_color}
    FIELD_SHAPES,{','.join([str(v) for v in field_shapes])}
    FIELD_LABELS,{','.join(field_labels)}
    FIELD_COLORS,{','.join(field_colors)}
    LEGEND_COLORS,{','.join(field_colors)}
    LEGEND_LABELS,{','.join(field_labels)}
    LEGEND_SHAPES,{','.join([str(v) for v in field_shapes])}
    LEGEND_TITLE,{legend_title}
    MARGIN,{margin}
    WIDTH,{width}
    BORDER_WIDTH,{border_width}
    SYMBOL_SPACING,{symbol_spacing}
    HEIGHT_FACTOR,{height_factor}
    DATA
    """)
    do_write(output_path, metadata, data)


def itol_color_strip_annotations(
    data: List[List[str]],
    output_path: Union[str, bytes, os.PathLike],
    dataset_label: str = 'Dataset',
    dataset_color: str = '#d55e00',
    legend_title: str = 'Color Strips',
    color_branches: bool = False,
    show_strip_label: bool = False,
    margin: int = 0,
    label_sorter: Callable[[str], int] = None,
) -> None:
    """
    Create iTOL color strip annotations.

    Example data input: 
    ```
    data = [
        ['GCA_001294575.1', '#ff0000', 'Label 1'],
        ['GCA_000761155.1', '#ffffff', 'Label 2'],
    ]
    ```

    - Format: 
        https://itol.embl.de/help.cgi#strip
    - Template: 
        https://itol.embl.de/help/dataset_color_strip_template.txt
    """
    if len(data) == 0:
        raise ValueError('Data is empty')

    n_elements = len(data[0]) - 1
    if n_elements <= 1:
        raise ValueError('Data elements must contain at least 3 elements')

    legend = {
        record[1]: record[2]
        for record in data
    }
    legend_colors = np.array([c for c in sorted(legend.keys())])
    legend_labels = [legend[c] for c in sorted(legend.keys())]

    sorted_tuples = sorted(
        zip(legend_labels, legend_colors), 
        key=lambda x: x[0] if label_sorter is None else label_sorter(x[0]),
    )
    a, b = zip(*sorted_tuples)
    legend_labels, legend_colors = list(a), list(b)

    metadata = textwrap.dedent(f"""\
    DATASET_COLORSTRIP
    SEPARATOR COMMA
    DATASET_LABEL,{dataset_label}
    COLOR,{dataset_color}
    COLOR_BRANCHES,{1 if color_branches else 0}
    SHOW_STRIP_LABELS,{1 if show_strip_label else 0}
    STRIP_LABEL_COLOR,#ffffff
    LEGEND_TITLE,{legend_title}
    LEGEND_COLORS,{','.join(legend_colors)}
    LEGEND_LABELS,{','.join(legend_labels)}
    LEGEND_SHAPES,{','.join(['1'] * len(legend_labels))}
    MARGIN,{margin}
    DATA
    """)
    do_write(output_path, metadata, data)


def itol_tree_color_annotations(
    data: List[List[str]],
    output_path: Union[str, bytes, os.PathLike],
) -> None:
    """
    Create iTOL tree color annotations.

    Example data input: 
    ```
    data = [
        ['GCA_001294575.1', 'label', '#0000ff'],
        ['GCA_001294575.1', 'label_background', 'rgba(255,0,0,0.5)'],
    ]
    ```

    - Format: 
        https://itol.embl.de/help.cgi#colors
    - Template: 
        https://itol.embl.de/help/colors_styles_template.txt
    """
    metadata = textwrap.dedent(f"""\
    TREE_COLORS
    SEPARATOR TAB
    DATA
    """)
    do_write(output_path, metadata, data, delimiter='\t')


def itol_shape_plot_annotations(
    data: List[List[str]],
    field_labels: List[str],
    field_colors: List[str],
    output_path: Union[str, bytes, os.PathLike],
    dataset_label: str = 'Shape plots',
    dataset_color: str = '#d55e00',
) -> None:
    """
    Create iTOL shape plot annotations.

    Example data input: 
    ```
    data = [
        ['GCA_001294575.1', 10, 10, 20, 40],
        ['GCA_000761155.1', 50, 60, 80, 90],
    ]
    ```

    - Format: 
        https://itol.embl.de/help.cgi#shapes
    - Template: 
        https://itol.embl.de/help/dataset_external_shapes_template.txt
    """
    metadata = textwrap.dedent(f"""\
    DATASET_EXTERNALSHAPE
    SEPARATOR COMMA
    DATASET_LABEL,{dataset_label}
    COLOR,{dataset_color}
    FIELD_COLORS,{','.join(field_colors)}
    FIELD_LABELS,{','.join(field_labels)}
    DASHED_LINES,1
    SHAPE_TYPE,2
    COLOR_FILL,1
    SHOW_INTERNAL,0
    HORIZONTAL_GRID,1
    VERTICAL_GRID,1
    LEGEND_TITLE,{dataset_label}
    LEGEND_SHAPES,{','.join(['2']*len(field_labels))}
    LEGEND_COLORS,{','.join(field_colors)}
    LEGEND_LABELS,{','.join(field_labels)}
    DATA
    """)
    do_write(output_path, metadata, data)


def itol_heatmap_annotations(
    data: List[List[str]],
    field_labels: List[str],
    output_path: Union[str, bytes, os.PathLike],
    dataset_label: str = 'Heatmap',
    dataset_color: str = '#d55e00',
    color_min: str = '#ffffff', 
    color_max: str = '#0173b2', 
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> None:
    """
    Create iTOL heatmap annotations.

    Example data input: 
    ```
    data = [
        ['GCA_001294575.1', 10, 10, 20, 40],
        ['GCA_000761155.1', 50, 60, 80, 90],
    ]
    ```

    - Format: 
        https://itol.embl.de/help.cgi#heatmap
    - Template: 
        https://itol.embl.de/help/dataset_heatmap_template.txt
    """
    if min_value is None:
        min_value = min([min(data[i][1:]) for i in range(len(data))])
    if max_value is None:
        max_value = max([max(data[i][1:]) for i in range(len(data))])

    metadata = textwrap.dedent(f"""\
    DATASET_HEATMAP
    SEPARATOR COMMA
    DATASET_LABEL,{dataset_label}
    COLOR,{dataset_color}
    FIELD_LABELS,{','.join(field_labels)}
    DASHED_LINES,1
    SHOW_INTERNAL,0
    LEGEND_TITLE,{dataset_label}
    LEGEND_LABELS,{','.join(field_labels)}
    COLOR_MIN,{color_min}
    COLOR_MAX,{color_max}
    USER_MIN_VALUE,{min_value}
    USER_MAX_VALUE,{max_value}
    DATA
    """)
    do_write(output_path, metadata, data)


def itol_bar_chart_annotations(
    data: List[List[str]],
    output_path: Union[str, bytes, os.PathLike],
    dataset_label: str = 'Bar chart',
    dataset_color: str = '#add8e6',
) -> None:
    metadata = textwrap.dedent(f"""\
    DATASET_SIMPLEBAR
    SEPARATOR COMMA
    DATASET_LABEL,{dataset_label}
    COLOR,{dataset_color}
    LEGEND_TITLE,{dataset_label}
    DATA
    """)
    do_write(output_path, metadata, data)


def itol_alignment_annotations(
    alignment_path: Union[str, bytes, os.PathLike],
    output_path: Union[str, bytes, os.PathLike],
    dataset_label: str = 'Alignment',
    dataset_color: str = '#d55e00',
    display_consensus: bool = True,
    display_conservation: bool = False,
) -> None:
    """
    Create iTOL alignment annotations.

    - Format: 
        https://itol.embl.de/help.cgi#alignment
    - Template: 
        https://itol.embl.de/help/dataset_alignment_template.txt
    """
    with open(alignment_path, 'rt') as f_in:
        for record in SeqIO.parse(alignment_path, 'fasta'):
            end_position = len(record.seq)
            break

    metadata = textwrap.dedent(f"""\
    DATASET_ALIGNMENT
    SEPARATOR COMMA
    DATASET_LABEL,{dataset_label}
    COLOR,{dataset_color}
    START_POSITION,1
    END_POSITION,{str(end_position)}
    DOTTED_DISPLAY,0
    DISPLAY_CONSENSUS,{'1' if display_consensus else '0'}
    DISPLAY_CONSERVATION,{'1' if display_conservation else '0'}
    DATA
    """)
    with open(alignment_path, 'rt') as f_in:
        with open(output_path, 'w') as f_out:
            f_out.write(metadata)
            for line in f_in:
                f_out.write(line)


def itol_protein_domain_annotations(
    data: List[List[str]],
    output_path: Union[str, bytes, os.PathLike],
    show_domain_label: bool = False,
    labels_on_top: bool = False,
    backbone_color='#aaaaaa',
    backbone_height=10,
    dataset_label: str = 'Potein domains',
    dataset_color: str = '#add8e6',
    margin : int = 5,
    label_sorter: Callable[[str], int] = None,
):
    """
    Create iTOL alignment annotations.

    - Format: 
        https://itol.embl.de/help.cgi#domains
    - Template: 
        https://itol.embl.de/help/dataset_protein_domains_template.txt

    There are 13 different possible shapes:
     - RE  rectangle
     - HH  horizontal hexagon
     - HV  vertical hexagon
     - EL  ellipse
     - DI  rhombus (diamond)
     - TR  right pointing triangle
     - TL  left pointing triangle
     - PL  left pointing pentagram
     - PR  right pointing pentagram
     - PU  up pointing pentagram
     - PD  down pointing pentagram
     - OC  octagon
     - GP  rectangle (gap; black filled rectangle with 1/3 normal height)

    Example: 
     - a 1200 amino acid protein with 3 domains, displayed on node 9606:
     - red colored rectangle representing a SH2 domain at positions 100 - 150
     - blue colored ellipse representing a SH3 domain at positions 400 - 500
     - green colored octagon representing a PH domain at positions 700 - 900
    
    9606,1200,RE|100|150|#ff0000|SH2,EL|400|500|#0000ff|SH3,OC|700|900|#00ff00|PH
    """
    legend_shapes = []
    legend_colors = []
    legend_labels = []
    seen_labels = set()
    for record in data:
        for domain_definition in record[2:]:
            shape, _, _, color, label = domain_definition.split('|')
            if label not in seen_labels:
                legend_labels.append(label)
                legend_colors.append(color)
                legend_shapes.append(shape)
                seen_labels.add(label)

    sorted_tuples = sorted(
        zip(legend_labels, legend_colors, legend_shapes), 
        key=lambda x: x[0] if label_sorter is None else label_sorter(x[0]),
    )
    a, b, c = zip(*sorted_tuples)
    legend_labels, legend_colors, legend_shapes = list(a), list(b), list(c)

    delimiter = '\t'
    metadata = textwrap.dedent(f"""\
    DATASET_DOMAINS
    SEPARATOR TAB
    DATASET_LABEL\t{dataset_label}
    COLOR\t{dataset_color}
    BACKBONE_COLOR\t{backbone_color}
    BACKBONE_HEIGHT\t{backbone_height}
    LEGEND_TITLE\t{dataset_label}
    LEGEND_SHAPES\t{delimiter.join(legend_shapes)}
    LEGEND_COLORS\t{delimiter.join(legend_colors)}
    LEGEND_LABELS\t{delimiter.join(legend_labels)}
    SHOW_DOMAIN_LABELS\t{'1' if show_domain_label else '0'}
    LABELS_ON_TOP\t{'1' if labels_on_top else '0'}
    MARGIN\t{margin}
    DATA
    """)
    do_write(output_path, metadata, data, delimiter=delimiter)


def itol_colored_ranges(
    data: List[List[str]],
    output_path: Union[str, bytes, os.PathLike],
    range_type: str,
    range_cover : str,
    dataset_label: str = 'colored range',
):
    """
    Colored/labeled ranges.

    https://itol.embl.de/help/dataset_ranges_template.txt

    Example
    a range between leaves 9606 and 184922, filled with a gradient from white (#ffffff) to red (#ff0000), 
    with a 2px dashed black (#000000) border and a blue (#0000ff) italic label:
    
    9606,184922,#ffffff,#ff0000,#000000,dashed,2,Example range,#0000ff,1,italic
    """
    if range_type not in ('box', 'bracket'):
        raise ValueError('Range type must be one of box or bracket')
    
    if range_cover not in ('label', 'clade', 'tree'):
        raise ValueError('Range cover must be one of label, clade or tree')
    
    metadata = textwrap.dedent(f"""\
    DATASET_RANGE
    SEPARATOR TAB
    DATASET_LABEL\t{dataset_label}
    COLOR\t#ffff00
    RANGE_TYPE\t{range_type}
    RANGE_COVER\t{range_cover}
    UNROOTED_SMOOTH\tsimplify
    COVER_LABELS\t0
    COVER_DATASETS\t0
    FIT_LABELS\t0
    BRACKET_STYLE\tsquare
    BRACKET_SIZE\t20
    BRACKET_SHIFT\t50
    BRACKET_BEHIND_DATASETS\t1
    SHOW_LABELS\t1
    LABEL_POSITION\tcenter-center
    LABELS_VERTICAL\t0
    STRAIGHT_LABELS\t0
    LABEL_ROTATION\t0
    LABEL_SHIFT_X\t0
    LABEL_SHIFT_Y\t0
    LABEL_OUTLINE_WIDTH\t0
    LABEL_OUTLINE_COLOR\t#ffffff
    LABEL_AUTO_COLOR\t1
    LABEL_SIZE_FACTOR\t1
    VERTICAL_SHRINK\t0
    DATA
    """)
    do_write(output_path, metadata, data, '\t')


def do_write(
    output_path : Union[str, bytes, os.PathLike],
    metadata : str,
    data : List[List[str]],
    delimiter : str = ',',
) -> None:
    with open(output_path, 'w') as f:
        f.write(metadata)
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerows(data)


def hex_to_rgba(hex_color, alpha=1.0):
    # Remove the hash symbol if it's there
    hex_color = hex_color.lstrip('#')
    
    # Convert hex values to integer RGB values
    red = int(hex_color[0:2], 16)
    green = int(hex_color[2:4], 16)
    blue = int(hex_color[4:6], 16)
    
    # Return as a tuple with the alpha value
    return f'rgba({red},{green},{blue},{alpha})'
