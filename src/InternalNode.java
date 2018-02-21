import java.util.*;
class InternalNode extends Node{
	String name;
	Map<Integer, Node> children;
	
	public InternalNode(String n) {
		name = n;
		children = new HashMap<>();
	}
	
	public void addChild(int attributeValue, Node n) {
		children.put(attributeValue, n);
	}
}
