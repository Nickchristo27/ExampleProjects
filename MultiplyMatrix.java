//Nicholas Christophides 113319835
import java.util.Scanner;
public class MultiplyMatrix {
	public static double[][] multiplyMatrix(double[][] a, double[][] b){
		int i, j;
		double[][] answer = new double[a.length][b[0].length];
		for (i = 0; i < answer.length; i++) {
			for (j = 0; j < answer[0].length; j++) {
				answer[i][j] = a[i][0]*b[0][j]+a[i][1]*b[1][j]+a[i][2]*b[2][j];
			}
		}
		return answer;
	}
	public static void main(String[] args) {
		System.out.println("Please input the first 3x3 matrix: ");
		Scanner input = new Scanner(System.in);
		double [][] a = new double[3][3];
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				a[i][j] = input.nextDouble();
			}
		}
		System.out.println("Please input the second 3x3 matrix: ");
		double [][] b = new double[3][3];
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				b[i][j] = input.nextDouble();
			}
		}
		input.close();
		double[][] answer = multiplyMatrix(a, b);
		System.out.println("Multiplication of the matrices is:");
		for (int i = 0; i < answer.length; i++) {
			for (int j = 0; j < answer[1].length; j++) {
				System.out.printf("%.1f ", answer[i][j]); 
			}
			System.out.println();
		}
	}
}
