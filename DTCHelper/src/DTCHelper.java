import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
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
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.JSONValue;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.parser.nndep.DependencyParser;
import edu.stanford.nlp.process.DocumentPreprocessor;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.semgrex.SemgrexPattern;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;

public class DTCHelper
{
	public static enum Mode{INIT, TRAIN, EVALUATE, CLASSIFY};
	public static final String addPatternCommand = "add_pattern";
	public static final String classifyCommand = "classify";
	public static final String endCommand = "end";
	public static final String hasPatternCommand = "has_pattern";
	public static final String parseCommand = "parse";
	public static final String setModeCommand = "set_mode";
	public static final String testCommand = "test";
	private static final String disconnectToken = "__DISCONNECT__";
	private static final String shutdownToken = "__SHUTDOWN__";
	private MaxentTagger tagger;
	private final DependencyParser parser;
	private final JSONParser jsonParser = new JSONParser();
	private final Map<Mode, Set<String>> validCommands = new HashMap<Mode, Set<String>>();
	private Map<String, Double> distribution = new HashMap<String, Double>();
	private Map<String, Set<String>> patternMap;
	private List<SemgrexPatternWrapper> semgrexPatterns;
	private Mode mode = Mode.INIT;
	
	public DTCHelper()
	{
		String modelFile = "edu/stanford/nlp/models/pos-tagger/english-left3words/"
			+ "english-left3words-distsim.tagger";
		
		tagger = new MaxentTagger(modelFile);
		parser = DependencyParser.loadFromModelFile(DependencyParser.DEFAULT_MODEL);
		
		validCommands.put(Mode.INIT, new HashSet<String>());
		validCommands.put(Mode.TRAIN, new HashSet<String>());
		validCommands.put(Mode.EVALUATE, new HashSet<String>());
		validCommands.put(Mode.CLASSIFY, new HashSet<String>());
		validCommands.get(Mode.INIT).add(endCommand);
		validCommands.get(Mode.INIT).add(setModeCommand);
		validCommands.get(Mode.TRAIN).add(addPatternCommand);
		validCommands.get(Mode.TRAIN).add(endCommand);
		validCommands.get(Mode.TRAIN).add(hasPatternCommand);
		validCommands.get(Mode.TRAIN).add(parseCommand);
		validCommands.get(Mode.TRAIN).add(setModeCommand);
		validCommands.get(Mode.EVALUATE).add(endCommand);
		validCommands.get(Mode.EVALUATE).add(setModeCommand);
		validCommands.get(Mode.EVALUATE).add(testCommand);
		validCommands.get(Mode.CLASSIFY).add(classifyCommand);
		validCommands.get(Mode.CLASSIFY).add(endCommand);
		validCommands.get(Mode.CLASSIFY).add(setModeCommand);
	}
	
	//Converts a sentence to a semantic graph (dependency tree).
	public SemanticGraph buildSemanticGraph(List<HasWord> sentence)
	{
		return new SemanticGraph(parser.predict(tagger.tagSentence(sentence)).typedDependencies());
	}
	
	//Classifies the given text.
	private String classifyText(String text)
	{
		SemanticGraph semanticGraph;
		String classLabel;
		
		distribution.clear();
		
		for(List<HasWord> sentence : new DocumentPreprocessor(new StringReader(text)))
		{
			semanticGraph = buildSemanticGraph(sentence);
			
			for(SemgrexPatternWrapper semgrexPatternWrapper : semgrexPatterns)
			{
				if(semgrexPatternWrapper.find(semanticGraph)
					&& verifyMatch(sentence, semgrexPatternWrapper))
				{
					if(!distribution.containsKey(semgrexPatternWrapper.getClassLabel()))
					{
						distribution.put(semgrexPatternWrapper.getClassLabel(), 0.0);
					}
					
					classLabel = semgrexPatternWrapper.getClassLabel();
					
					distribution.put(classLabel, distribution.get(classLabel) + 1.0);
					break;
				}
			}
		}
		
		return JSONObject.toJSONString(distribution);
	}
	
	//Converts the given text to a JSON object containing the string representation of a dependency
	//tree.
	private String parseText(String text)
	{
		List<String> sentences = new ArrayList<String>();
		String formatted;
		
		for(List<HasWord> sentence : new DocumentPreprocessor(new StringReader(text)))
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
	
	//Executes a command given by the dependency tree classifier client.
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
					
					if(!patternMap.containsKey(classValue))
					{
						patternMap.put(classValue, new HashSet<String>());
					}
					
					patternMap.get(classValue).add((String) jsonObject.get("pattern"));
				}
				else if(command.equals(classifyCommand))
				{
					return classifyText((String) jsonObject.get("text"));
				}
				else if(command.equals(endCommand))
				{
					return disconnectToken;
				}
				else if(command.equals(hasPatternCommand))
				{
					String classLabel = (String) jsonObject.get("class");
					boolean hasPattern = patternMap.containsKey(classLabel)
						&& patternMap.get(classLabel).contains((String) jsonObject.get("pattern"));
					return JSONValue.toJSONString(hasPattern);
				}
				else if(command.equals(parseCommand))
				{
					return parseText((String) jsonObject.get("text"));
				}
				else if(command.equals(setModeCommand))
				{
					setMode((String) jsonObject.get("mode"));
				}
				else if(command.equals(testCommand))
				{
					testPatterns((String) jsonObject.get("text"), (String) jsonObject.get("class"));
				}
			}
		}
		
		return null;
	}
	
	//Sets the current mode and performs actions necessary for transition.
	public void setMode(Mode mode)
	{
		this.mode = mode;
		
		switch(mode)
		{
			case INIT:
				break;
			case TRAIN:
				semgrexPatterns = null;
				patternMap = new HashMap<String, Set<String>>();
				break;
			case EVALUATE:
				int count = 0;
				Map<String, Integer> classCounts = new HashMap<String, Integer>();
				
				for(Entry<String, Set<String>> entry : patternMap.entrySet())
				{
					classCounts.put(entry.getKey(), entry.getValue().size());
					
					count += entry.getValue().size();
				}
				
				semgrexPatterns = new ArrayList<SemgrexPatternWrapper>(count);
				
				for(String classValue : patternMap.keySet())
				{
					for(String pattern : patternMap.get(classValue))
					{
						SemgrexPattern semgrexPattern = SemgrexPattern.compile(pattern);
						
						semgrexPatterns.add(new SemgrexPatternWrapper(semgrexPattern, classValue));
					}
				}
				
				patternMap = null;
				break;
			case CLASSIFY:
				Collections.sort(semgrexPatterns);
				break;
			default:
				return;
		}
		
		System.out.println("Mode set to " + mode.toString() + ".");
	}
	
	//Sets the current mode based on a string value.
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
	
	//Tests semgrex patterns on the given text and class label.
	private void testPatterns(String text, String classLabel)
	{
		SemanticGraph semanticGraph;
		
		for(List<HasWord> sentence : new DocumentPreprocessor(new StringReader(text)))
		{
			semanticGraph = buildSemanticGraph(sentence);
			
			for(SemgrexPatternWrapper semgrexPatternWrapper : semgrexPatterns)
			{
				semgrexPatternWrapper.test(semanticGraph, classLabel);
			}
		}
	}
	
	//Ensures that a semgrex pattern with multiple occurrences of the same word matches a sentence
	//that has the same number of occurrences of that word.
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
		    InputStream inputStream = clientSocket.getInputStream();
		    BufferedReader in = new BufferedReader(new InputStreamReader(inputStream));
		    String inputLine, outputLine, client = clientSocket.getInetAddress().getHostAddress();
		    
		    System.out.println("Connected to " + client + ".");
		    
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
