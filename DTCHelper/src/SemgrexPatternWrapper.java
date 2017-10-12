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
		
		while((matcher = regex.matcher(pattern)).find())
		{
			word = pattern.substring(matcher.start());
			word = word.substring(word.indexOf(startToken) + startToken.length(), word.indexOf(endToken));
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
	public int compareTo(SemgrexPatternWrapper semgrexPatternWrapper)
	{
		double accuracyA = getAccuracy(), accuracyB = semgrexPatternWrapper.getAccuracy();
		return accuracyA < accuracyB ? 1 : accuracyA > accuracyB ? -1 : 0;
	}
	
	public boolean equals(SemgrexPatternWrapper semgrexPatternWrapper)
	{
		return semgrexPattern.pattern().equals(semgrexPatternWrapper.semgrexPattern.pattern()) && classLabel.equals(semgrexPatternWrapper.getClassLabel());
	}
	
	public boolean find(SemanticGraph semanticGraph)
	{
		return semgrexPattern.matcher(semanticGraph).find();
	}
	
	public double getAccuracy()
	{
		if(accuracy < 0.0)
		{
			accuracy = 0.0;
			
			double correctClass, incorrectClass;
			Set<String> classes = new HashSet<String>(correct.keySet());
			
			classes.addAll(incorrect.keySet());
			
			for(String classLabel : classes)
			{
				correctClass = correct.containsKey(classLabel) ? correct.get(classLabel) : 0.0;
				incorrectClass = incorrect.containsKey(classLabel) ? incorrect.get(classLabel) : 0.0;
				accuracy += correctClass / (correctClass + incorrectClass);
			}
			
			accuracy /= (double) classes.size();
			correct = null;
			incorrect = null;
		}
		
		return accuracy;
	}
	
	public String getClassLabel()
	{
		return classLabel;
	}
	
	public List<String> getWords()
	{
		return words;
	}
	
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
	
	public String toString()
	{
		return semgrexPattern.pattern();
	}
}
