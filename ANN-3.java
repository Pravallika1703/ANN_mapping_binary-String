
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Scanner;

public class ANN
{
float[][] inhid,hidout;
float bias=0.0f,o_bias=0.0f;
int[] inputlayer,target;
float[] outputlayer,hidden;
float n=0.1f;
String[] bstring={"10000000","01000000","00100000","00010000","00001000","00000100",
		"00000010","00000001"};

// Assigning random weights
	public void build()
	{
		inhid=new float[8][3];
		hidout=new float[3][8];
		hidden=new float[3];
		outputlayer=new float[8];
		inputlayer=new int[8];
		target=new int[8];
		for(int i=0;i<8;i++)
		{
			for(int j=0;j<3;j++)
			{
				inhid[i][j]=(float)Math.random()-(float)0.5;
			}
		}

	for(int i=0;i<3;i++)
		{
			for(int j=0;j<8;j++)
			{
				hidout[i][j]=(float)Math.random()-(float)0.5;

			}}


	}
	// training neural network selecting random inputs and weights.. final weight
	//vector is stored in inhid and hidout matrices
	public void train(int numIterations)
	{
		//System.out.println("in train");
		float h = 0,op=0,p=0;
		float deltah[]=new float[3];
		float deltak[]=new float[8];
		while(numIterations>0)
		{

			 int index=(int) Math.floor(8*Math.random());
			 String input=bstring[index];
			 for(int i=0;i<input.length();i++)
				{
					inputlayer[i]=Character.getNumericValue(input.charAt(i));
					target[i]=inputlayer[i];


				}

			for(int i=0;i<3;i++)
			{
				h=0;
				for(int j=0;j<8;j++)
				{
					 h=h+inputlayer[j]*inhid[j][i];
				}
				h=h+bias*1;
				hidden[i]=(float) ((float)1/(1+Math.exp(-h)));
				//System.out.println("hidden   "+ i+ "      value: "+hidden[i]);
			}
			for(int i=0;i<8;i++)
			{
				op=0;
				for(int j=0;j<3;j++)
				{
					op=op+hidden[j]*hidout[j][i];
				}
				op=op+o_bias*1;
				outputlayer[i]=(float) ((float)1/(1+Math.exp(-op)));

				//float d=(float) ((float)1/(1+Math.exp(-2)));
				//System.out.println("test"+d);
				//System.out.println("output   "+ i+ "value: "+outputlayer[i]);

			}
				 for(int j=0;j<8;j++)
				 {
					 deltak[j]=outputlayer[j]*(1-outputlayer[j])*(target[j]-outputlayer[j]);
					// System.out.println("deltak"+ deltak[j]+"target"+target[j]);
				 }


			 for(int i=0;i<3;i++)
			 {
				 p=0;
				 for(int j=0;j<8;j++)
				 {
              p=p+hidout[i][j]*deltak[j];
				 }
				 deltah[i]=hidden[i]*(1-hidden[i])*p;
			 }
			 //System.out.println("weighit vector input and hidden");

			 for(int i=0;i<3;i++)
			 {
				 for(int j=0;j<8;j++)
				 {
					 hidout[i][j]=hidout[i][j]+(n*deltak[j]*hidden[i]);
				//	 System.out.print(hidout[i][j]);
				 }
				 //System.out.println();

			 }
			 //System.out.println("weighit vector hidden and output");

			 for(int i=0;i<8;i++)
			 {
				 for(int j=0;j<3;j++)
				 {
					 inhid[i][j]=inhid[i][j]+(n*deltah[j]*inputlayer[i]);
				//	 System.out.print(inhid[i][j]);

				 }
				 //
				 //System.out.println();

			 }

	     numIterations--;
		}

	}
	public String fit(String input)
	{
		float h=0,op=0;
		 for(int i=0;i<input.length();i++)
			{
				inputlayer[i]=Character.getNumericValue(input.charAt(i));


			}

		for(int i=0;i<3;i++)
		{
			h=0;
			for(int j=0;j<8;j++)
			{
				 h=h+inputlayer[j]*inhid[j][i];
			}
			h=h+bias*1;
			hidden[i]=(float) ((float)1/(1+Math.exp(-h)));
			System.out.println("hidden   "+ i+ "value:  "+hidden[i]);
		}
		String str="";
		for(int i=0;i<8;i++)
		{
			op=0;
			for(int j=0;j<3;j++)
			{
				op=op+hidden[j]*hidout[j][i];
			}
			op=op+o_bias*1;
			outputlayer[i]=(float) ((float)1/(1+Math.exp(-op)));
			if(outputlayer[i]>=(float)0.5)
				outputlayer[i]=1;
			else
				outputlayer[i]=0;
		str=str+String.valueOf((int)outputlayer[i])	;
		}

		return str;
	}
	public void save() throws IOException
	{
		FileOutputStream fos = new FileOutputStream("anet.txt");
	      ObjectOutputStream oos = new ObjectOutputStream(fos);
	      oos.writeObject(inhid);
	      oos.writeObject(hidout);
    }
	public void restore() throws IOException, ClassNotFoundException
	{
		FileInputStream fin = new FileInputStream("anet.txt");
	      ObjectInputStream oos = new ObjectInputStream(fin);
	      inhid=(float[][]) oos.readObject();
	      hidout=(float[][]) oos.readObject();


	}
	public static void main(String args[])
	{
		System.out.println("Training neural network, wait until it finishes");

		ANN an=new ANN();
		an.build();
		an.train(5000000);
		System.out.println("Finished training");

		System.out.println("Enter input string for fit function");
		Scanner sc=new Scanner(System.in);
		String iput=sc.next();
		String s=an.fit(iput);
		System.out.println("Output string:    "+s);

	}
}
