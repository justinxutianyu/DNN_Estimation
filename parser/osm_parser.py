
# Import necessary packages.
import networkx as nx
import xml.etree.ElementTree as ET
import re

def osm_xml_parser(path_to_file):
    """Function to parse an osm file and create a network out of it.

    Parameters:
        filename - The filename of the file to import.
    Returns:
        graph - The created graph.
    """

    # Parse the xml structure and initialize variables.
    #e = ET.parse(path_to_file).getroot()
    with open(path_to_file) as f:
        file = f.read()
    e = ET.fromstring(re.sub(r"(<\?xml[^>]+\?>)", r"\1<root>", file) + "</root>")

    node_dict_tmp = {}
    G = nx.DiGraph()

    # Allow these types of streets to be represented in the network by an edge.
    way_types = ["motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential", "service",
                 "living_street"]

    # Create nodes and edges.
    for i in e:
        # Nodes.
        if i.tag == "node":
            node_dict_tmp[i.attrib["id"]] = [i.attrib["lat"], i.attrib["lon"]]

        # Edges.
        if i.tag == "way":
            insert = False
            directed = False
            max_speed_v = None
            way_tmp = []
            for j in i:
                if j.tag == "nd":
                    way_tmp.append(j.attrib["ref"])
                if j.tag == "tag":
                    if j.attrib["k"] == "oneway" and j.attrib["v"] == "yes":
                        directed = True
                    if j.attrib["k"] == "highway" and j.attrib["v"] in way_types:
                        insert = True
                    if j.attrib["k"] == "maxspeed":
                        try:
                            max_speed_v = (float(j.attrib["v"]) * 1000) / 3600
                            if max_speed_v <= 0 or max_speed_v > 25:
                                max_speed_v = None
                        except:
                            max_speed_v = None
            if insert:
                if max_speed_v is None:
                    G.add_path(way_tmp)
                    if not directed:
                        G.add_path(list(reversed(way_tmp)))
                else:
                    G.add_path(way_tmp, max_speed=max_speed_v)
                    if not directed:
                        G.add_path(list(reversed(way_tmp)), max_speed=max_speed_v)

    # Extend the nodes by their geographical coordinates.
    network_nodes = G.nodes()
    for i in network_nodes:
        current_node_coords = node_dict_tmp[i]
        G.node[i]["coords"] = [float(current_node_coords[0]), float(current_node_coords[1])]

    # Return the generated graph.
    return G

# Specify the path to the OSM-XML file and call the parser.
path_to_file = "parser_test.osm"#"hawaii-latest.osm"
graph = osm_xml_parser(path_to_file)