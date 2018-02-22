import java.io.*;
import java.util.*;

public class DecisionTree {
	
	public static void main(String[] args) throws Exception {
		
		String trainingSet1 = "src/data_sets2/training_set.csv";
		String validationSet1 = "src/data_sets2/validation_set.csv";
		String testSet1 = "src/data_sets2/test_set.csv";
		Scanner scanner;
		int l = 10;
		int k = 10;
		String toPrint = "yes";
		
		System.out.println("Enter the 6 parameters");

		BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(System.in));
		List<String> parameters = Arrays.asList(bufferedReader.readLine().split(" "));

		if(parameters.size()!=6) {
			System.out.println("Not the required number of parameters. Exiting.");
			return;
		}
		
		l = Integer.parseInt(parameters.get(0));
		k = Integer.parseInt(parameters.get(1));
		trainingSet1 = parameters.get(2);
		validationSet1 = parameters.get(3);
		testSet1 = parameters.get(4);
		toPrint = parameters.get(5);
		
		
		scanner = new Scanner(new File(trainingSet1));
		
		List<Map<String,Integer>> trainingDataSet = new ArrayList<>();
		List<String> attributes = Arrays.asList(scanner.nextLine().split(","));
		
		Map<String,Integer> dataPoint;
		
		while(scanner.hasNext()) {
			dataPoint = new HashMap<>();
			List<String> attributeValues = Arrays.asList(scanner.nextLine().split(","));
			int i = 0;
			String attributeName = null;
			for(String value: attributeValues) {
				attributeName = attributes.get(i);
				dataPoint.put(attributeName, Integer.parseInt(value));
				i++;
			}
			i=0;
			trainingDataSet.add(dataPoint);
		}
		
		scanner.close();
		
		//check if any two points are identical and remove one of them
		for(int i = 0; i<trainingDataSet.size(); i++) {
			for(int j = i + 1; j<trainingDataSet.size(); j++) {
				if(areDatapointsIdentical(trainingDataSet.get(i), trainingDataSet.get(j))) {
					trainingDataSet.remove(j);
				}
			}
		}
		
		Node headNodeInfoGain = growTree(trainingDataSet, "INFORMATION_GAIN");
		Node headNodeVarianceImpurity = growTree(trainingDataSet, "VARIANCE_IMPURITY_HEURISTIC");
		
		//Validation dataset
		scanner = new Scanner(new File(validationSet1));
		List<Map<String, Integer>> validationDataSet = new ArrayList<>();
		attributes = Arrays.asList(scanner.nextLine().split(","));
		
		while(scanner.hasNext()) {
			dataPoint = new HashMap<>();
			List<String> attributeValues = Arrays.asList(scanner.nextLine().split(","));
			int i = 0;
			String attributeName = null;
			for(String value: attributeValues) {
				attributeName = attributes.get(i);
				dataPoint.put(attributeName, Integer.parseInt(value));
				i++;
			}
			i=0;
			validationDataSet.add(dataPoint);
		}
		Node prunedNode = postPrune(headNodeInfoGain, l, k, validationDataSet);
		Node prunedNodeVarianceImpurity = postPrune(headNodeVarianceImpurity, l, k, validationDataSet);
		//Test dataset
		scanner = new Scanner(new File(testSet1));
		List<Map<String, Integer>> testDataSet = new ArrayList<>();
		attributes = Arrays.asList(scanner.nextLine().split(","));
		
		while(scanner.hasNext()) {
			dataPoint = new HashMap<>();
			List<String> attributeValues = Arrays.asList(scanner.nextLine().split(","));
			int i = 0;
			String attributeName = null;
			for(String value: attributeValues) {
				attributeName = attributes.get(i);
				dataPoint.put(attributeName, Integer.parseInt(value));
				i++;
			}
			i=0;
			testDataSet.add(dataPoint);
		}
		
		double accuracyOfTestSetUsingInformationGain = testDataSet(headNodeInfoGain, testDataSet);
		System.out.println("Test data accuracy using Information Gain: " + accuracyOfTestSetUsingInformationGain);
		double accuracyOfTestSetUsingInformationGainOnPrunedTree = testDataSet(prunedNode, testDataSet);
		System.out.println("Test data accuracy using Information Gain on pruned tree: " + accuracyOfTestSetUsingInformationGainOnPrunedTree);
		
		double accuracyOfTestSetUsingVarianceImpurity = testDataSet(headNodeVarianceImpurity, testDataSet);
		System.out.println("Test data accuracy using Variance Impurity: " + accuracyOfTestSetUsingVarianceImpurity);
		double accuracyOfTestSetUsingVarianceImpurityOnPrunedTree = testDataSet(prunedNodeVarianceImpurity, testDataSet);
		System.out.println("Test data accuracy using Variance Impurity on pruned tree: " + accuracyOfTestSetUsingVarianceImpurityOnPrunedTree);
		
		if(toPrint.toLowerCase().equals("yes")) {
			System.out.println("Pre pruned tree using Information Gain: ");
			printTree(headNodeInfoGain, 0);
			System.out.println("\nPost pruned tree using Information Gain: ");
			printTree(prunedNode, 0);
			
			System.out.println("\nPre pruned tree using Variance Impurity: ");
			printTree(headNodeVarianceImpurity, 0);
			System.out.println("\nPost pruned tree using VarianceImpurity: ");
			printTree(prunedNodeVarianceImpurity, 0);
		}
		scanner.close();
	}
	
	public static boolean areDatapointsIdentical(Map<String, Integer> dp1, Map<String, Integer> dp2) {
		boolean identical = true;
		for(String key: dp1.keySet()) {
			if(!key.equals("Class")) {
				if(dp1.get(key) != dp2.get(key)) {
					identical = false;
					break;
				}
			}
		}
		return identical;
	}
	
	public static Node postPrune(Node node, int l, int k, List<Map<String, Integer>> validationDataSet) {
		Node bestNode = node;
		Node clonedTree;
		double bestAccuracy = testDataSet(node, validationDataSet);
		for(int i = 0; i<l; i++) {
			clonedTree = cloneTree(node);
			Random rand = new Random();
			int m = rand.nextInt(k) + 1;
			for(int j = 0; j<m; j++) {
				List<Node> nonLeafNodes = listOfNonLeafNodes(clonedTree);
				int n = nonLeafNodes.size();
				if (n>0) {
					int p = rand.nextInt(n) + 1;
					Node pNode = nonLeafNodes.get(p - 1);
					//perform pruning
					List<Integer> classesAtPNode = categoriesAtPNode(pNode);
					int val = (classesAtPNode.get(0) > classesAtPNode.get(1)) ? 0 : 1;
					if (pNode == clonedTree) {
						clonedTree = new Leaf(val);
					} else {
						Queue<Node> queue = new LinkedList<>();
						queue.add(clonedTree);
						while (!queue.isEmpty()) {
							Node current = queue.poll();
							if (current instanceof InternalNode) {
								for (int o : ((InternalNode) current).children.keySet()) {
									if (((InternalNode) current).children.get(o) == pNode) {
										((InternalNode) current).children.put(o, new Leaf(val));
									} else {
										queue.add(((InternalNode) current).children.get(o));
									}
								}
							}
						}
					} 
				}
				
			}
			double currentAccuracy = testDataSet(clonedTree, validationDataSet);
			if(currentAccuracy>bestAccuracy) {
				bestAccuracy = currentAccuracy;
				bestNode = clonedTree;
			}
		}
		return bestNode;
	}
	public static List<Integer> categoriesAtPNode(Node node){
		int zeroes = countVals(node, 0);
		int ones = countVals(node, 1);
		
		List<Integer> answer = new ArrayList<>();
		
		answer.add(zeroes);
		answer.add(ones);
		
		return answer;
	}
	
	public static int countVals(Node node, int val) {
		if(node instanceof Leaf && ((Leaf) node).category == val) {
			return 1;
		}
		else if(node instanceof InternalNode) {
			return countVals(((InternalNode) node).children.get(0), val) + countVals(((InternalNode) node).children.get(1), val);
		}
		return 0;
	}
	
	public static Node cloneTree(Node node) {
		Node clonedNode;
		if(node instanceof Leaf) {
			clonedNode = new Leaf(((Leaf) node).category);
		}
		else {
			clonedNode = new InternalNode(((InternalNode) node).name);
			Node temp;
			for(int i: ((InternalNode) node).children.keySet()) {
				temp = ((InternalNode) node).children.get(i);
				Node temp2 = cloneTree(temp);
				((InternalNode)clonedNode).addChild(i, temp2);
			}
		}
		return clonedNode;
	}
	
	public static List<Node> listOfNonLeafNodes(Node node){
		List<Node> nonLeafNodes = new ArrayList<>();
		Queue<Node> queue = new LinkedList<>();
		queue.add(node);
		while(!queue.isEmpty()) {
			Node temp = queue.poll();
			if(temp instanceof InternalNode) {
				nonLeafNodes.add(temp);
				for(int i: ((InternalNode) temp).children.keySet()) {
					queue.add(((InternalNode) temp).children.get(i));
				}
			}
		}
		return nonLeafNodes;
	}
	
	public static Node pruneNode(Node pNode) {
		List<Integer> classesAtPNode = categoriesAtPNode(pNode);
		if(classesAtPNode.get(0)>classesAtPNode.get(1)) {
			pNode = new Leaf(0);
		}
		else {
			pNode = new Leaf(1);
		}
		
		return pNode;
	}
	
	public static Node parentNode(Node node) {
		return null;
	}
	public static double testDataSet(Node node, List<Map<String, Integer>> dataSet) {
		int truePositives = 0;
		int trueNegatives = 0;
		int falsePositives = 0;
		int falseNegatives = 0;
		double accuracy = 0;
		
		for(Map<String, Integer>dataPoint: dataSet) {
			int classFromTree = performTreeTraversal(node, dataPoint);
			if(classFromTree == 1 && dataPoint.get("Class") == 1) {
				truePositives++;
			}
			else if(classFromTree == 1 && dataPoint.get("Class") == 0) {
				falsePositives++;
			}
			else if(classFromTree == 0 && dataPoint.get("Class") == 0) {
				trueNegatives++;
			}
			else if(classFromTree == 0 && dataPoint.get("Class") == 1) {
				falseNegatives++;
			}
		}
		
		accuracy = ((double)(truePositives + trueNegatives)) / ((double)(truePositives + trueNegatives + falsePositives + falseNegatives));
		
		return accuracy;
	}
	
	public static int performTreeTraversal(Node node, Map<String, Integer> dataPoint) {
		int classFromTree = -1;
		if(node instanceof Leaf) {
			return ((Leaf) node).category;
		}
		else if(node instanceof InternalNode) {
			if(dataPoint.get(((InternalNode) node).name) == 0) {
				return performTreeTraversal(((InternalNode) node).children.get(0), dataPoint);
			}
			else if(dataPoint.get(((InternalNode) node).name) == 1) {
				return performTreeTraversal(((InternalNode) node).children.get(1), dataPoint);
			}
		}
		return classFromTree;
	}
	
	public static void printTree(Node node, int level) {
		
		if(node instanceof Leaf) {
			System.out.print(((Leaf)node).category);
		}
		else if(node instanceof InternalNode) {
			System.out.println();
			for(int i = 0; i<level; i++) {
				System.out.print("| ");
			}
			InternalNode temp = (InternalNode)node;
			System.out.print(temp.name + " = 0 : ");
			printTree(temp.children.get(0), level + 1);
			System.out.println();
			for(int i = 0; i<level; i++) {
				System.out.print("| ");
			}
			System.out.print(temp.name + " = 1 : ");
			printTree(temp.children.get(1), level + 1);
		}
		
	}
	
	public static Node growTree(List<Map<String,Integer>> dataSet, String method) {
		Node current;
		int size = dataSet.size();
		
		int zeroClassCount = 0;
		int oneClassCount = 0;
		for(Map<String,Integer> dataPoint: dataSet) {
			if(dataPoint.get("Class") == 0) {
				zeroClassCount++;
			}
			else if(dataPoint.get("Class") == 1) {
				oneClassCount++;
			}
			else {
				System.out.println("Anomaly Class:" + dataPoint.get("Class"));
			}
		}
		
		if(zeroClassCount == size) {
			current = new Leaf(0);
		}
		
		else if(oneClassCount == size) {
			current = new Leaf(1);
		}
		
		else {
			String bestAttribute = bestGainAttribute(dataSet, method);

			List<Map<String, Integer>> s0 = new ArrayList<>(), s1 = new ArrayList<>();
			
			for(Map<String, Integer> dataPoint: dataSet) {
				if(dataPoint.get(bestAttribute) == 0) {
					s0.add(dataPoint);
				}
				else if(dataPoint.get(bestAttribute) == 1) {
					s1.add(dataPoint);
				}
				else {
					System.out.println("anomaly: bestAttribute" + bestAttribute + "value: " + dataPoint.get(bestAttribute));
				}
			}
			if(s0.size() == dataSet.size()) {
				System.out.println("dds");
			}
			if(s1.size() == dataSet.size()) {
				System.out.println("ddss");
			}
			Node nodeForZero = growTree(s0, method);
			Node nodeForOne = growTree(s1, method);
			
			current = new InternalNode(bestAttribute);
			current.addChild(0,nodeForZero);
			current.addChild(1, nodeForOne);
			
		}
		
		return current;
	}
	
	public static String bestGainAttribute(List<Map<String,Integer>> dataSet, String method) {
		String bestAttribute = "";

		double initialEntropy = initialEntropy(dataSet);
		
		double maxInformationGain = 0;
		
		Set<String> attributesSet = new HashSet<>();
		for(String attr: dataSet.get(0).keySet()) {
			if(!attr.equals("Class")) {
				attributesSet.add(attr);
			}
		}
		
		for(String attribute: attributesSet) {
			
			double informationGain = 0;
			switch(method) {
			case "INFORMATION_GAIN":
				informationGain = informationGainForAttribute(dataSet, initialEntropy, attribute);
				break;
			case "VARIANCE_IMPURITY_HEURISTIC":
				informationGain = varianceImpurityGainForAttribute(dataSet, initialEntropy, attribute);
				break;
			}
			
			if(informationGain>maxInformationGain) {
				maxInformationGain = informationGain;
				bestAttribute = attribute;
			}
		}

		return bestAttribute;
	}
	
	public static double varianceImpurityGainForAttribute(List<Map<String,Integer>> dataSet, double initialEntropy, String attribute) {
		int total = dataSet.size();
		int attributeZero = 0;
		int attributeOne = 0;
		
		
		List<Map<String, Integer>> s0 = new ArrayList<>(), s1 = new ArrayList<>();
		
		for(Map<String, Integer> dataPoint: dataSet) {
			if(dataPoint.get(attribute) == 0) {
				attributeZero++;
				s0.add(dataPoint);
			}
			else if(dataPoint.get(attribute) == 1) {
				attributeOne++;
				s1.add(dataPoint);
			}
			

		}
		
		double varianceImpurity = varianceImpurity(dataSet);
		double zeroVarianceImpurity = varianceImpurity(s0);
		double oneVarianceImpurity = varianceImpurity(s1);
		double zeroProbability = ((double)(attributeZero)/(double)total);
		double oneProbability = ((double)(attributeOne)/(double)total);
		double gain = varianceImpurity - ((zeroProbability*zeroVarianceImpurity + oneProbability*oneVarianceImpurity));
		
		return gain;
	}
	
	public static double varianceImpurity(List<Map<String, Integer>> dataSet) {
		if(dataSet.size() == 0)
			return 0;
		double varianceImpurity = 0;
		int classZero = 0;
		int classOne = 0;
		for(Map<String, Integer> dataPoint: dataSet) {
			if(dataPoint.get("Class") == 0) {
				classZero++;
			}
			else if(dataPoint.get("Class") == 1) {
				classOne++;
			}
		}
		varianceImpurity = calculateVarianceImpurity(classZero, classOne);
		return varianceImpurity;
	}
	
	public static double calculateVarianceImpurity(int zeroes, int ones) {
		if(zeroes == 0 || ones == 0)
			return 0;
		return ((double)(zeroes*ones))/((double)((zeroes + ones)*(zeroes + ones)));
	}
	
	
	private static double informationGainForAttribute(List<Map<String, Integer>> dataSet, double initialEntropy,
			String attribute) {
		int zeroZero = 0;
		int zeroOne = 0;
		int oneZero = 0;
		int oneOne = 0;
		
		for(Map<String, Integer> dataPoint: dataSet) {
			if(dataPoint.get(attribute) == 0) {
				if(dataPoint.get("Class") == 0) {
					zeroZero++;
				}
				else if(dataPoint.get("Class") == 1) {
					zeroOne++;
				}
			}
			else if(dataPoint.get(attribute) == 1) {
				if(dataPoint.get("Class") == 0) {
					oneZero++;
				}
				else if(dataPoint.get("Class") == 1) {
					oneOne++;
				}
			}
		}
		
		double entropyLeft = calculateEntropy(zeroZero, zeroOne);
		double entropyRight = calculateEntropy(oneZero, oneOne);
		double informationGain = informationGain(initialEntropy, zeroZero, zeroOne, oneZero, oneOne, entropyLeft,
				entropyRight);
		return informationGain;
	}

	private static double informationGain(double initialEntropy, int zeroZero, int zeroOne, int oneZero, int oneOne,
			double entropyLeft, double entropyRight) {
		int total = zeroZero + zeroOne + oneZero + oneOne;
		double finalEntropy = ((double)((double)zeroZero + (double)zeroOne)/((double)total))*(entropyLeft) + ((double)((double)oneZero + (double)oneOne)/((double)total))*(entropyRight);
		double informationGain = initialEntropy - finalEntropy;
		return informationGain;
	}

	private static double initialEntropy(List<Map<String, Integer>> dataSet) {
		double initialEntropy = 0;
		int zeroClassCount = 0;
		int oneClassCount = 0;
		
		for(Map<String, Integer> dataPoint: dataSet) {
			if(dataPoint.get("Class") == 0) {
				zeroClassCount++;
			}
			else if(dataPoint.get("Class") == 1) {
				oneClassCount++;
			}
			else {
				System.out.println("anomaly: " + dataPoint.get("Class"));
			}
		}
		
		initialEntropy = calculateEntropy(zeroClassCount, oneClassCount);
		return initialEntropy;
	}
	
	public static double calculateEntropy(int zeroCount, int oneCount) {
		double entropy = 0;
		double zeroFraction = 0;
		double oneFraction = 0;
		double zeroFractionLog = 0;
		double oneFractionLog = 0;
		
		if(zeroCount != 0) {
			zeroFraction = (double)((double)(zeroCount))/((double)(zeroCount + oneCount));
			zeroFractionLog = Math.log(zeroFraction)/Math.log((double)2);
		}
		
		if(oneCount != 0) {
			oneFraction = (double)((double)(oneCount))/((double)(zeroCount + oneCount));
			oneFractionLog = Math.log(oneFraction)/Math.log((double)2);
		}
		
		entropy = -(zeroFraction*zeroFractionLog + oneFraction*oneFractionLog);
		return entropy;
	}
	
}
