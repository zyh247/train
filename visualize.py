""" Network architecture visualizer using graphviz """
import sys
from graphviz import Digraph
import genotypes as gt


def plot(genotype, file_path, caption=None):
    """ make DAG plot and save to file_path as .png """
    edge_attr = {
        'fontsize': '20',
        'fontname': 'Arial'
    }
    node_attr = {
        'style': 'filled',
        'shape': 'rect',
        'align': 'center',
        'fontsize': '20',
        'height': '0.5',
        'width': '0.5',
        'penwidth': '2',
        'fontname': 'Arial'
    }
    g = Digraph(
        format='png',
        edge_attr=edge_attr,
        node_attr=node_attr,
        engine='dot')
    g.body.extend(['rankdir=LR'])

    # input nodes
    g.node("c_{k-2}", fillcolor='darkseagreen2')
    g.node("c_{k-1}", fillcolor='darkseagreen2')

    # intermediate nodes
    n_nodes = 4  # Based on the range(2, 6) in normal_concat
    for i in range(n_nodes):
        g.node(str(i), fillcolor='lightblue')

    # 连接操作，正确处理节点连接
    assert len(genotype) % 2 == 0
    steps = len(genotype) // 2

    # 为每个中间节点创建两个输入连接
    for i in range(steps):
        for j in range(2):
            op, idx = genotype[i * 2 + j]
            if idx == 0:
                u = "c_{k-2}"
            elif idx == 1:
                u = "c_{k-1}"
            else:
                u = str(idx - 2)

            v = str(i)
            g.edge(u, v, label=op, fillcolor="gray")

    # output node
    g.node("c_{k}", fillcolor='palegoldenrod')
    for i in range(n_nodes):
        g.edge(str(i), "c_{k}", fillcolor="gray")

    # add image caption
    if caption:
        g.attr(label=caption, overlap='false', fontsize='20', fontname='Arial')

    # 确保文件路径不包含.png扩展名，因为render函数会自动添加
    if file_path.endswith('.png'):
        file_path = file_path[:-4]

    try:
        g.render(file_path, view=False, cleanup=True)
        print(f"成功生成图片: {file_path}.png")
    except Exception as e:
        print(f"生成图片时发生错误: {e}")
        # 尝试使用不同的引擎
        try:
            g = Digraph(
                format='png',
                edge_attr=edge_attr,
                node_attr=node_attr,
                engine='neato')
            g.body.extend(['rankdir=LR'])
            print("尝试使用neato引擎重新生成...")
            g.render(file_path, view=False, cleanup=True)
            print(f"成功生成图片: {file_path}.png")
        except Exception as e2:
            print(f"使用备用引擎时也发生错误: {e2}")


if __name__ == '__main__':
    # 图片中显示的CIFAR-10上学习到的正常单元结构
    normal_cell = [('sep_conv_3x3', 0), ('avg_pool_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 3),
                ('dil_conv_3x3', 1), ('avg_pool_3x3', 3), ('avg_pool_3x3', 0)]

    # 图片中显示的第二个结构（使用max_pool_3x3）
    pooling_cell = [('skip_connect', 0), ('skip_connect', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('dil_conv_3x3', 2),
                ('sep_conv_3x3', 1), ('sep_conv_5x5', 3), ('sep_conv_5x5', 0)]


    genotype = gt.Genotype(
        normal=normal_cell,
        normal_concat=range(2, 6),
        reduce=pooling_cell,
        reduce_concat=range(2, 6)
    )

    # 生成图片
    plot(genotype.normal, "../network_visualization/normal.png", caption="Normal cell learned on CIFAR-10.")
    plot(genotype.reduce, "../network_visualization/reduction.png", caption="Pooling cell learned on CIFAR-10.")
