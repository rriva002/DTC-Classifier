import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.semgrex.SemgrexPattern;

public class SemgrexPatternWrapper implements Comparable<SemgrexPatternWrapper>
{
	private SemgrexPattern semgrexPattern;
	private String classLabel;
	private Map<String, Double> correct = new HashMap<String, Double>();
	private Map<String, Double> incorrect = new HashMap<String, Double>();
	private List<String> words = new ArrayList<String>();
	private double accuracy = -1.0;
	
	public SemgrexPatternWrapper(SemgrexPattern semgrexPattern, String classLabel)
	{
		String pattern = semgrexPattern.pattern(), word, startToken = "/", endToken = ".*/}";
		Pattern regex = Pattern.compile("\\{word:/.*\\.\\*/\\}");
		Matcher matcher;
		int startIndex;
		
		while((matcher = regex.matcher(pattern)).find())
		{
			word = pattern.substring(matcher.start());
			startIndex = word.indexOf(startToken) + startToken.length();
			word = word.substring(startIndex, word.indexOf(endToken));
			pattern = pattern.substring(matcher.start() + word.length() + endToken.length());
			
			words.add(word);
		}
		
		if(words.size() == new HashSet<String>(words).size())
		{
			words = new ArrayList<String>(0);
		}
		
		this.semgrexPattern = semgrexPattern;
		this.classLabel = classLabel;
	}
	
	@Override
	//Comparison function for sorting by weighted accuracy.
	public int compareTo(SemgrexPatternWrapper semgrexPatternWrapper)
	{
		double accuracyA = getAccuracy(), accuracyB = semgrexPatternWrapper.getAccuracy();
		return accuracyA < accuracyB ? 1 : accuracyA > accuracyB ? -1 : 0;
	}
	
	//Determines whether the semgrex pattern and associated class label are identical to the given
	//semgrex pattern and associated class label.
	public boolean equals(SemgrexPatternWrapper semgrexPatternWrapper)
	{
		return semgrexPattern.pattern().equals(semgrexPatternWrapper.semgrexPattern.pattern())
			&& classLabel.equals(semgrexPatternWrapper.getClassLabel());
	}
	
	//Determines whether the semgrex pattern matches part of the given semantic graph (dependency
	//tree).
	public boolean find(SemanticGraph semanticGraph)
	{
		return semgrexPattern.matcher(semanticGraph).find();
	}
	
	//Returns the weighted accuracy of the semgrex pattern.
	public double getAccuracy()
	{
		if(accuracy < 0.0)
		{
			accuracy = 0.0;
			
			double correctClass, incorrectClass;
			Set<String> classLabels = new HashSet<String>(correct.keySet());
			
			classLabels.addAll(incorrect.keySet());
			
			for(String label : classLabels)
			{
				correctClass = correct.containsKey(label) ? correct.get(label) : 0.0;
				incorrectClass = incorrect.containsKey(label) ? incorrect.get(label) : 0.0;
				accuracy += correctClass / (correctClass + incorrectClass);
			}
			
			accuracy /= (double) classLabels.size();
			correct = null;
			incorrect = null;
		}
		
		return accuracy;
	}
	
	//Returns the class label associated with the semgrex pattern.
	public String getClassLabel()
	{
		return classLabel;
	}
	
	//Returns the list of words in the semgrex pattern.
	public List<String> getWords()
	{
		return words;
	}
	
	//Tests the semgrex pattern on the given semantic graph (dependency tree).
	public void test(SemanticGraph semanticGraph, String classLabel)
	{
		boolean matched = find(semanticGraph), sameClass = classLabel.equals(this.classLabel);
		
		if((matched && sameClass) || (!matched && !sameClass))
		{
			if(!correct.containsKey(classLabel))
			{
				correct.put(classLabel, 0.0);
			}
			
			correct.put(classLabel, correct.get(classLabel) + 1.0);
		}
		else
		{
			if(!incorrect.containsKey(classLabel))
			{
				incorrect.put(classLabel, 0.0);
			}
			
			incorrect.put(classLabel, incorrect.get(classLabel) + 1.0);
		}
	}
	
	//Returns the string representation of the semgrex pattern.
	public String toString()
	{
		return semgrexPattern.pattern();
	}
}
