import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.io.StringReader;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.SocketException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Queue;
import java.util.Set;
import java.util.function.Predicate;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.JSONValue;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.ling.Word;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import edu.stanford.nlp.parser.nndep.DependencyParser;
import edu.stanford.nlp.process.DocumentPreprocessor;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.semgrex.SemgrexPattern;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;
import edu.stanford.nlp.trees.Tree;

public class DTCHelper
{
	public static enum Mode{INIT, TRAIN, EVALUATE, CLASSIFY};
	public static final String addPatternCommand = "add_pattern";
	public static final String classifyCommand = "classify";
	public static final String endCommand = "end";
	public static final String hasPatternCommand = "has_pattern";
	public static final String parseCommand = "parse";
	public static final String setModeCommand = "set_mode";
	public static final String splitSentencesCommand = "split_sentences";
	public static final String testCommand = "test";
	private static final String disconnectToken = "__DISCONNECT__";
	private static final String shutdownToken = "__SHUTDOWN__";
	private static final String sbar = "SBAR";
	private final MaxentTagger tagger = new MaxentTagger("edu/stanford/nlp/models/pos-tagger/english-left3words/english-left3words-distsim.tagger");
	private final DependencyParser parser = DependencyParser.loadFromModelFile(DependencyParser.DEFAULT_MODEL);
	private final LexicalizedParser constituencyParser = LexicalizedParser.loadModel();
	private final JSONParser jsonParser = new JSONParser();
	private final Map<Mode, Set<String>> validCommands = new HashMap<Mode, Set<String>>(Mode.values().length);
	private Map<String, Double> distribution = new HashMap<String, Double>();
	private Map<String, Set<String>> semgrexPatternMap;
	private List<SemgrexPatternWrapper> semgrexPatterns;
	private Mode mode = Mode.INIT;
	private boolean splitSentences = false;
	private static final Predicate<Tree> predicate = new Predicate<Tree>()
	{
		@Override
		public boolean test(Tree tree)
		{
			return !tree.value().equals(sbar);
		}
	};
	
	public DTCHelper()
	{
		validCommands.put(Mode.INIT, new HashSet<String>());
		validCommands.put(Mode.TRAIN, new HashSet<String>());
		validCommands.put(Mode.EVALUATE, new HashSet<String>());
		validCommands.put(Mode.CLASSIFY, new HashSet<String>());
		validCommands.get(Mode.INIT).add(endCommand);
		validCommands.get(Mode.INIT).add(setModeCommand);
		validCommands.get(Mode.INIT).add(splitSentencesCommand);
		validCommands.get(Mode.TRAIN).add(addPatternCommand);
		validCommands.get(Mode.TRAIN).add(endCommand);
		validCommands.get(Mode.TRAIN).add(hasPatternCommand);
		validCommands.get(Mode.TRAIN).add(parseCommand);
		validCommands.get(Mode.TRAIN).add(setModeCommand);
		validCommands.get(Mode.TRAIN).add(splitSentencesCommand);
		validCommands.get(Mode.EVALUATE).add(endCommand);
		validCommands.get(Mode.EVALUATE).add(setModeCommand);
		validCommands.get(Mode.EVALUATE).add(testCommand);
		validCommands.get(Mode.CLASSIFY).add(classifyCommand);
		validCommands.get(Mode.CLASSIFY).add(endCommand);
		validCommands.get(Mode.CLASSIFY).add(setModeCommand);
	}
	
	public SemanticGraph buildSemanticGraph(List<HasWord> sentence)
	{
		return new SemanticGraph(parser.predict(tagger.tagSentence(sentence)).typedDependencies());
	}
	
	private String classifyText(String text, String classLabel)
	{
		SemanticGraph semanticGraph;
		
		distribution.clear();
		
		for(List<HasWord> sentence : splitSentences ? parseSentences(text) : new DocumentPreprocessor(new StringReader(text)))
		{
			semanticGraph = buildSemanticGraph(sentence);
			
			for(SemgrexPatternWrapper semgrexPatternWrapper : semgrexPatterns)
			{
				if(semgrexPatternWrapper.find(semanticGraph) && verifyMatch(sentence, semgrexPatternWrapper))
				{
					if(!distribution.containsKey(semgrexPatternWrapper.getClassLabel()))
					{
						distribution.put(semgrexPatternWrapper.getClassLabel(), 0.0);
					}
					
					distribution.put(semgrexPatternWrapper.getClassLabel(), distribution.get(semgrexPatternWrapper.getClassLabel()) + 1.0);
					break;
				}
			}
		}
		
		return JSONObject.toJSONString(distribution);
	}
	
	private List<Tree> parseClauses(Tree root)
	{
		Queue<Tree> queue = new LinkedList<Tree>();
		List<Tree> trees = new ArrayList<Tree>();
		String rootValue = root.value();
		Tree tree;
		
		for(Tree child : root.children())
		{
			if(!child.value().equals(sbar) && !child.value().equals("S"))
			{
				queue.add(child);
			}
		}
		
		while(!queue.isEmpty())
		{
			root.removeChild(root.objectIndexOf(queue.remove()));
		}
		
		queue.add(root);
		
		while(!queue.isEmpty())
		{
			tree = queue.remove();
			
			for(Tree child : tree.children())
			{
				if(child.value().equals(sbar))
				{
					trees.addAll(parseClauses(child));
				}
				else
				{
					queue.add(child);
				}
			}
		}
		
		root.setValue("ROOT");
		
		tree = root.prune(predicate);
		
		if(tree != null)
		{
			trees.add(tree.deepCopy());
		}
		
		root.setValue(rootValue);
		return trees;
	}
	
	private List<List<HasWord>> parseSentences(String text)
	{
		List<List<HasWord>> sentences = new ArrayList<List<HasWord>>();
		StringBuilder stringBuilder = new StringBuilder(text.length());
		
		for(List<HasWord> sentence : new DocumentPreprocessor(new StringReader(text)))
		{
			for(Tree tree : parseClauses(constituencyParser.parse(sentence)))
			{
				stringBuilder.delete(0, stringBuilder.length());
				
				for(Word word : tree.yieldWords())
				{
					stringBuilder.append(stringBuilder.length() > 0 ? " " : "");
					stringBuilder.append(word.word());
				}
				
				for(List<HasWord> clause : new DocumentPreprocessor(new StringReader(stringBuilder.toString())))
				{
					sentences.add(clause);
				}
			}
		}
		
		return sentences;
	}
	
	private String parseText(String text)
	{
		List<String> sentences = new ArrayList<String>();
		String formatted;
		
		for(List<HasWord> sentence : splitSentences ? parseSentences(text) : new DocumentPreprocessor(new StringReader(text)))
		{
			formatted = buildSemanticGraph(sentence).toFormattedString().replace("\n", " ");
			
			while(formatted.contains("  "))
			{
				formatted = formatted.replace("  ", " ");
			}
			
			sentences.add(formatted);
		}
		
		return JSONArray.toJSONString(sentences);
	}
	
	public String receiveCommand(String json) throws IOException, ParseException
	{
		JSONObject jsonObject = (JSONObject) jsonParser.parse(json);
		String commandString = "command";
		
		if(jsonObject.containsKey(commandString))
		{
			String command = (String) jsonObject.get(commandString);
			
			if(validCommands.get(mode).contains(command))
			{
				if(command.equals(addPatternCommand))
				{
					String classValue = (String) jsonObject.get("class");
					
					if(!semgrexPatternMap.containsKey(classValue))
					{
						semgrexPatternMap.put(classValue, new HashSet<String>());
					}
					
					semgrexPatternMap.get(classValue).add((String) jsonObject.get("pattern"));
				}
				else if(command.equals(classifyCommand))
				{
					return classifyText((String) jsonObject.get("text"), (String) jsonObject.get("class"));
				}
				else if(command.equals(endCommand))
				{
					return disconnectToken;
				}
				else if(command.equals(hasPatternCommand))
				{
					String classValue = (String) jsonObject.get("class");
					return JSONValue.toJSONString(semgrexPatternMap.containsKey(classValue) && semgrexPatternMap.get(classValue).contains((String) jsonObject.get("pattern")));
				}
				else if(command.equals(parseCommand))
				{
					return parseText((String) jsonObject.get("text"));
				}
				else if(command.equals(setModeCommand))
				{
					setMode((String) jsonObject.get("mode"));
				}
				else if(command.equals(splitSentencesCommand))
				{
					setSplitSentences((Boolean) jsonObject.get("value"));
				}
				else if(command.equals(testCommand))
				{
					testSemgrexPatterns((String) jsonObject.get("text"), (String) jsonObject.get("class"));
				}
			}
		}
		
		return null;
	}
	
	public void setMode(Mode mode)
	{
		this.mode = mode;
		
		switch(mode)
		{
			case INIT:
				break;
			case TRAIN:
				semgrexPatterns = null;
				semgrexPatternMap = new HashMap<String, Set<String>>();
				break;
			case EVALUATE:
				int count = 0;
				Map<String, Integer> classCounts = new HashMap<String, Integer>();
				
				for(Entry<String, Set<String>> entry : semgrexPatternMap.entrySet())
				{
					classCounts.put(entry.getKey(), entry.getValue().size());
					
					count += entry.getValue().size();
				}
				
				semgrexPatterns = new ArrayList<SemgrexPatternWrapper>(count);
				
				for(String classValue : semgrexPatternMap.keySet())
				{
					for(String pattern : semgrexPatternMap.get(classValue))
					{
						semgrexPatterns.add(new SemgrexPatternWrapper(SemgrexPattern.compile(pattern), classValue));
					}
				}
				
				semgrexPatternMap = null;
				break;
			case CLASSIFY:
				Collections.sort(semgrexPatterns);
				break;
			default:
				return;
		}
		
		System.out.println("Mode set to " + mode.toString() + ".");
	}
	
	public void setMode(String modeValue)
	{
		if(modeValue.equals("train"))
		{
			setMode(Mode.TRAIN);
		}
		else if(modeValue.equals("evaluate"))
		{
			setMode(Mode.EVALUATE);
		}
		else if(modeValue.equals("classify"))
		{
			setMode(Mode.CLASSIFY);
		}
	}
	
	public void setSplitSentences(boolean splitSentences)
	{
		this.splitSentences = splitSentences;
	}
	
	private void testSemgrexPatterns(String text, String classLabel)
	{
		SemanticGraph semanticGraph;
		
		for(List<HasWord> sentence : splitSentences ? parseSentences(text) : new DocumentPreprocessor(new StringReader(text)))
		{
			semanticGraph = buildSemanticGraph(sentence);
			
			for(SemgrexPatternWrapper semgrexPatternWrapper : semgrexPatterns)
			{
				semgrexPatternWrapper.test(semanticGraph, classLabel);
			}
		}
	}
	
	private boolean verifyMatch(List<HasWord> sentence, SemgrexPatternWrapper semgrexPatternWrapper)
	{
		StringBuilder stringBuilder = new StringBuilder(sentence.size() * 5);
		String wordToken = "__WORD_TOKEN__";
		
		for(HasWord word : sentence)
		{
			stringBuilder.append(wordToken).append(word.word()).append(" ");
		}
		
		String sentenceText = stringBuilder.toString();
		
		for(String word : semgrexPatternWrapper.getWords())
		{
			if(!sentenceText.contains(wordToken + word))
			{
				return false;
			}
			
			sentenceText = sentenceText.replaceFirst(wordToken + word, "");
		}
		
		return true;
	}
	
	public static void main(String[] args) throws IOException, ParseException
	{
		int port = 9000;
		DTCHelper dtcHelper = new DTCHelper();
		ServerSocket serverSocket = new ServerSocket(port);
		
		System.out.println("Listening on port " + port + ".");
		
		while(true)
		{
			Socket clientSocket = serverSocket.accept();
		    PrintWriter out = new PrintWriter(clientSocket.getOutputStream(), true);
		    BufferedReader in = new BufferedReader(new InputStreamReader(clientSocket.getInputStream()));
		    String inputLine, outputLine;
		    
		    System.out.println("Connected to " + clientSocket.getInetAddress().getHostAddress() + ".");
		    
		    try
		    {
			    while((inputLine = in.readLine()) != null)
			    {
			    	outputLine = dtcHelper.receiveCommand(inputLine);
			    	
			    	if(outputLine != null)
			    	{
			    		if(outputLine.equals(disconnectToken))
			    		{
			    			break;
			    		}
			    		else if(outputLine.equals(shutdownToken))
			    		{
			    			clientSocket.close();
			    			serverSocket.close();
			    			return;
			    		}
			    		
			    		out.println(outputLine.replace("\n", "__NEWLINE__"));
			    	}
			    }
		    }
		    catch(SocketException socketException)
		    {
		    	System.out.println("Disconnected from client.");
		    }
		    
		    clientSocket.close();
		}
	}
}
