
import os

import networkx as nx

from natsort        import natsorted
from dataclasses    import dataclass 

from data.utils import (
    load_graph_from_json, 
    visualize_application, 
    get_compute_list_from_json)

@dataclass
class DependOn:
    id                 : int
    end_cycle          : int

@dataclass
class Node:
    id                  : int
    type                : str

    processing_time     : int = None
    generate            : int = None

    minimum_start_cycle : int = None
    actual_start_cylce  : int = None
    end_cycle           : int = None

    depend_list         : list[DependOn] = None
    is_assigned         : bool = False

PACKET_SIZE = 4 # flits

def get_list_of_successor_nodes_as_node_obj(graph: nx.DiGraph, node_id: int) -> list:
    """
    Returns a list of Node objects that are successors to the node_id.  
    The Node object has the following attributes:  (only the relevant attributes are mentioned)
        1. Depend_list is assigned with the dependent nodes  
        2. minimum_start_cycle is assigned (to PACKET_SIZE * require) if the successor node has no other dependencies  
    """

    # Assign minimum_wait_time to all the successor nodes
    
    list_of_successor_nodes = []
    successor_nodes = list(graph.successors(node_id))
    
    for successor_node in successor_nodes:
        require             = graph.edges[node_id, successor_node]['weight']
        minimum_start_cycle = PACKET_SIZE * require

        # Check if the successor node has any other dependency
        dependent_list = []
        has_other_dependency = False
        predecessor_nodes    = list(graph.predecessors(successor_node))

        for predecessor_node in predecessor_nodes:

            if graph.nodes[predecessor_node]['type'] != "dependency":

                depend = DependOn(id = predecessor_node, end_cycle = None)
                has_other_dependency = True
                dependent_list.append(depend)

        if has_other_dependency:
            minimum_start_cycle = None

        # Creating a node object
        node = Node(id = successor_node,
                    type                = graph.nodes[successor_node]['type'],
                    processing_time     = graph.nodes[successor_node]['processing_time'],
                    generate            = graph.nodes[successor_node]['generate'],
                    minimum_start_cycle = minimum_start_cycle,
                    actual_start_cylce  = None, 
                    end_cycle           = None,
                    depend_list         = dependent_list)

        list_of_successor_nodes.append(node)

    return list_of_successor_nodes

def get_non_dependency_nodes_count(graph: nx.DiGraph) -> int:

    non_dependency_nodes_count = 0
    for idx, node in graph.nodes(data=True):
        if node["type"] != "dependency":
            non_dependency_nodes_count += 1

    return non_dependency_nodes_count

def init_first_dependency_node( graph: nx.DiGraph ) -> tuple[list, int]:
    """
    Find the dependecy node (assuming there is only one dependency node)  
    Iterate through the successor nodes and assign minimum_start_cycle (if conditions are met)  

        Note: Conditions to assign minimum_start_cycle, 
            1. If the successor to the `dependency` node has no other   
                dependencirs than the `dependency` node. 

    Returns   
        1. a list of Node objects, these are successor nodes of the `dependency` node.  
        2. Index of the node (of type 'task_depend') with the least minimum_start_cycle and no other dependencies.
    """

    # Assign minimum_wait_time to all applicable dependent nodes
    # and finding the node that will start the earliest
    for idx, node in graph.nodes(data=True):

        if node["type"] == "dependency":

            successor_nodes         = list(graph.successors(idx))

            list_of_successor_nodes = get_list_of_successor_nodes_as_node_obj(graph, idx)

            # Finding the node with least minimum_start_cycle
            # to assign the actual_start_cycle
            start_cycle_node_idx_value = (None, None) # (index, start_cycle)
            not_initialized_flag = True

            for idx, node in enumerate(list_of_successor_nodes):

                if len(node.depend_list) != 0:
                    # You can also do node.minimum_start_cycle is None
                    continue

                if not_initialized_flag:
                    not_initialized_flag        = False
                    start_cycle_node_idx_value  = (idx, node.minimum_start_cycle)

                if node.minimum_start_cycle < start_cycle_node_idx_value[1]:
                    start_cycle_node_idx_value = (idx, node.minimum_start_cycle)

            # Assigning the actual_start_cycle
            list_of_successor_nodes[start_cycle_node_idx_value[0]].actual_start_cylce = start_cycle_node_idx_value[1]

            assert list_of_successor_nodes[start_cycle_node_idx_value[0]].actual_start_cylce == list_of_successor_nodes[start_cycle_node_idx_value[0]].minimum_start_cycle

    return list_of_successor_nodes, start_cycle_node_idx_value[0]
    
def update_end_cylce(executed_node: Node) -> Node: 
    """Computing the end cycle with processing time and generate and actual_start_cycle"""
    executed_node.end_cycle = executed_node.actual_start_cylce + ( executed_node.processing_time * executed_node.generate ) 
    executed_node.is_assigned = True

    return executed_node

def update_depend_on_end_cycle_for_all_nodes(node_list: list[Node], executed_node:Node) -> None:
    """ 
    Updates the DependOn.end_cycle for all the nodes in the node_list
    that are dependent on the executed_node """

    for node in node_list:
        if node.is_assigned:
            continue

        for depend in node.depend_list:
            if depend.id == executed_node.id:
                depend.end_cycle = executed_node.end_cycle


def check_if_all_successor_nodes_in_node_list(node_list: list[Node]) -> list[Node]:
    """Check if the nodes in the node_list have all their successors in the node_list"""
    new_node_list = []

    for node in node_list:
        has_successor, successor_nodes = check_if_node_has_successor(graph, node.id)

        if has_successor:
            for successor_node in successor_nodes:

                if not is_node_in_node_list(node_list, successor_node):

                    if not is_node_in_node_list(new_node_list, successor_node):
                        list_of_new_nodes = get_list_of_successor_nodes_as_node_obj(graph, node.id)
                        new_node_list.extend(list_of_new_nodes)

    return new_node_list


def update_depend_on_for_new_node_list(node_list: list[Node], new_node_list: list[Node]) -> None:    
    """
    For the new_node_list, 
    update the DependOn.end_cycle with the end_cycle of the node it is dependent on.
    """

    for new_node in new_node_list: 

        for depend in new_node.depend_list:
            depend_id = depend.id

            for node in node_list:

                if not node.is_assigned:
                    continue
                if node.id == depend_id:
                    depend.end_cycle = node.end_cycle


def node_list_has_all_nodes(node_list: list[Node], num_nodes: int) -> bool:
    if len(node_list) != num_nodes:
        return False
    return True

def check_all_nodes_assigned(node_list: list[Node]) -> bool:
    
    for node in node_list:
        if not node.is_assigned:
            return False
    return True

def check_if_node_has_successor(graph: nx.DiGraph, node_id: int) -> tuple[bool, list]:

    successor_nodes = list(graph.successors(node_id))
    if len(successor_nodes) == 0:
        return False, []

    return True, successor_nodes

def is_node_in_node_list(node_list: list[Node], node_id: int) -> bool:

    for node in node_list:
        if node.id == node_id:
            return True

    return False

def assign_minimum_start_cycle_as_depend_cycle(node_list: list[Node]) -> None:
    """ 
    For nodes that have full dependency list [DependOn.end_cycle != None],    
    Assign the minimum start_cycle to the DependOn.end_cycle """
    for node in node_list:

        if node.is_assigned:
            continue

        all_depend_have_end_cycle = False

        for depend in node.depend_list:

            if depend.end_cycle is None:
                all_depend_have_end_cycle = False
                break

            all_depend_have_end_cycle = True

        if all_depend_have_end_cycle:
            # Assign the maximum end_cycle of the dependecy list to the minimum_start_cycle
            node.minimum_start_cycle = max([depend.end_cycle for depend in node.depend_list])


def get_node_with_least_minimum_start_cycle(node_list: list[Node]) -> Node:

    minimum_index = None
    first_flag = True

    for idx, node in enumerate(node_list):

        if node.is_assigned:
            continue

        if node.minimum_start_cycle is None:
            continue

        if first_flag:
            minimum_index = idx

        if node.minimum_start_cycle < node_list[minimum_index].minimum_start_cycle:
            minimum_index = idx    

    print(f"\nNode list ")
    for node in node_list:
        print(node)

    print(f"Starting node {node_list[minimum_index]}")


    return node_list[minimum_index]


def get_list_nodes_ready_to_execute(node_list: list[Node]) -> list[Node]: 

    list_of_nodes_ready_to_execute = []
    for node in node_list:

        if node.is_assigned:
            continue

        all_depend_have_end_cycle = False

        if len(node.depend_list) != 0:
            for depend in node.depend_list:

                if depend.end_cycle is None:
                    all_depend_have_end_cycle = False
                    break

                all_depend_have_end_cycle = True

        else: 
            all_depend_have_end_cycle = True

        if all_depend_have_end_cycle:
            list_of_nodes_ready_to_execute.append(node)

    return list_of_nodes_ready_to_execute
    

def assign_cycle_for_executing_node(node_list: list[Node], node_id: int) -> None:

    # Finding max end cycle of all the assigned nodes
    list_of_start_cycles = []
    for node in node_list:

        if not node.is_assigned:
            continue

        list_of_start_cycles.append(node.end_cycle)

    max_end_cycle_of_assigned_nodes = max(list_of_start_cycles)


    # Finding the max of dependency end cycle
    list_of_depend_end_cycles = []
    for node in node_list:
        if node.id == node_id:
            for depend in node.depend_list:
                for node in node_list:
                    if node.id == depend.id:
                        list_of_depend_end_cycles.append(node.end_cycle)
    

    if len(list_of_depend_end_cycles) == 0:
        start_cycle = max_end_cycle_of_assigned_nodes
    else: 
        max_end_cycle_of_depend_nodes = max(list_of_depend_end_cycles)
        start_cycle = max(max_end_cycle_of_assigned_nodes, max_end_cycle_of_depend_nodes)

    for node in node_list:
        if node.id == node_id:
            node.actual_start_cylce = start_cycle + 1


def get_minimum_start_cycle_from_assigned_nodes(node_list: list[Node]) -> int:

    list_of_start_cycles = []

    for node in node_list:

        if not node.is_assigned:
            continue

        list_of_start_cycles.append(node.end_cycle)

    return max(list_of_start_cycles)


def convert_node_list_to_compute_dict(node_list: list[Node]) -> dict:

    compute_dict = {}
    for node in node_list:
        compute_dict[node.id] = {
            "start_cycle": node.actual_start_cylce,
            "end_cycle": node.end_cycle
        }

    return compute_dict


def main(graph):
    
    # Finding the first node to execute
    non_dependency_nodes_count  = get_non_dependency_nodes_count(graph)
    node_list, start_node_index = init_first_dependency_node(graph)
    start_node                  = node_list[start_node_index]
    executed_node               = update_end_cylce(start_node) 

    update_depend_on_end_cycle_for_all_nodes(node_list, executed_node)

    # Finding the next node to execute
    while True:   

        if node_list_has_all_nodes(node_list, non_dependency_nodes_count):
            if check_all_nodes_assigned(node_list):
                break

        new_node_list = check_if_all_successor_nodes_in_node_list(node_list)
        update_depend_on_for_new_node_list(node_list, new_node_list)

        node_list.extend(new_node_list)

        assign_minimum_start_cycle_as_depend_cycle(node_list)

        nodes_ready_to_execute          = get_list_nodes_ready_to_execute(node_list)
        start_node                      = get_node_with_least_minimum_start_cycle(nodes_ready_to_execute)
        possible_minimum_start_cycle    = get_minimum_start_cycle_from_assigned_nodes(node_list)

        if possible_minimum_start_cycle > start_node.minimum_start_cycle:
            start_node.actual_start_cylce = possible_minimum_start_cycle + 1

        else: 
            start_node.actual_start_cylce = start_node.minimum_start_cycle + 1

        update_end_cylce(start_node)    
        update_depend_on_end_cycle_for_all_nodes(node_list, start_node)


    return node_list


PACKET_SIZE = 4 # flits

if __name__ == "__main__":

    import sys

    if len(sys.argv) > 1:   
        use_analytical_test_data    = sys.argv[1].lower() in ['true', '1']

    else:                   
        use_analytical_test_data    = False

    if use_analytical_test_data:
        input_data_path     = "data/analytical_test_data/input"
        target_path         = "data/analytical_test_data/target"

    else: 
        input_data_path     = "data/training_data/test/input"
        target_path         = "data/training_data/test/target"

    input_data_files    = natsorted( os.listdir( input_data_path ) )
    target_files        = natsorted( os.listdir( target_path ) )

    for input_idx, target_idx in zip(input_data_files, target_files):

        graph           = load_graph_from_json( 
                            os.path.join(input_data_path, input_idx))

        schedule_list   = get_compute_list_from_json( 
                            os.path.join(target_path, target_idx))

        dependency_node_count = 0

        for idx, node in graph.nodes(data=True):
            if node["type"] == "dependency":
                dependency_node_count += 1

        # Skip Graphs with more than 1 dependency node
        if dependency_node_count > 1:
            continue
        
        node_list = main(graph)

        analytical_schedule_list = convert_node_list_to_compute_dict(node_list)
        
        visualize_application(graph,  compute_list=schedule_list, pred_compute_list=analytical_schedule_list)

