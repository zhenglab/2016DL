－


#include "MLP.h"

//∂®“Â∏ﬂÀπÀÊª˙ ˝(∑¬–¥”·¿œ ¶deep.cpp÷–µƒ∏ﬂÀπÀÊª˙ ˝∫Ø ˝)
float Parameter::sample_from_gaussian(float miu, float sigma){
	static float V1, V2, S;
	static int phase = 0;
	float X;
	float gaussian_output;
	if (phase == 0){
		do{
			float U1 = (float)rand() / RAND_MAX;
			float U2 = (float)rand() / RAND_MAX;
			V1 = 2 * U1 - 1;
			V2 = 2 * U2 - 1;
			S = V1 * V1 + V2 * V2;
		} while (S >= 1 || S == 0);
		X = V1 * sqrt(-2 * log(S) / S);
	}
	else
		X = V2 * sqrt(-2 * log(S) / S);
	phase = 1 - phase;
	gaussian_output = X * sigma + miu;
	return gaussian_output;
}

//∂®“Âº§ªÓ∫Ø ˝sigmoid
float Functions::sigmoid(float x){
	float Y;
	Y = 1 / (1 + exp(-x));
	return Y;
}

//∂®“Âsigmoid∫Ø ˝µƒµº∫Ø ˝
float Functions::gradsigmoid(float x){
	float G;
	G = x*(1 - x);
	return G;
}
//∂®“Â«∞¿°Õ¯¬Á^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
float feedforward(Inputlayer &inp, Node_value &nva,Parameter &par){
	
	Functions fun;
	float a, b, c, sum;
	float loss;
	
	for (int n = 0; n < data_number; n++)
	{
		a = 0;
		sum = 0;
		b = 0;
		c = 0;
		//Õ®π˝—≠ª∑«ÛΩ‚√ø≤„Ω⁄µ„µƒ÷µ
		for (int i = 0; i<first_hiden_layer_node_number; i++) {//µ⁄“ª“˛≤ÿ≤„Ω⁄µ„ ‰≥ˆ÷µ£ª
			a = 0;
			for (int j = 0; j<input_layer_node_number; j++) {
				a += par.ih_weight_array[j][i] * inp.input_data[n][j];
			}
			nva.first_hiden_layer_node_value[n][i] = fun.sigmoid(a + par.first_hiden_layer_bias_array[i]);
		}
		
		for (int i = 0; i<second_hiden_layer_node_number; i++) {//µ⁄∂˛“˛≤ÿ≤„Ω⁄µ„ ‰≥ˆ÷µ£ª
			b = 0;
			for (int j = 0; j<first_hiden_layer_node_number; j++) {
				b += par.hh_weight_array[j][i] * nva.first_hiden_layer_node_value[n][j];
			}
			nva.second_hiden_layer_node_value[n][i] = fun.sigmoid(b + par.second_hiden_layer_bias_array[i]);
		}
		for (int i = 0; i<output_layer_node_number; i++) {// ‰≥ˆ≤„Ω⁄µ„¿€º”÷µ£ª
			c = 0;
			for (int j = 0; j<second_hiden_layer_node_number; j++) {
				c += par.ho_weight_array[j][i] * nva.second_hiden_layer_node_value[n][j];
			}
			sum += exp(c);
			
		}
		
		for (int i = 0; i<output_layer_node_number; i++) {// ‰≥ˆ≤„Ω⁄µ„æ≠softmax∫Ø ˝∫Ûµƒ ‰≥ˆ÷µ£ª
			c = 0;
			for (int j = 0; j<second_hiden_layer_node_number; j++) {
				c += par.ho_weight_array[j][i] * nva.second_hiden_layer_node_value[n][j];
			}
			nva.output_layer_node_value[n][i] = exp(c) / sum;
			if (nva.output_layer_node_value[n][i] != nva.output_layer_node_value[n][i])
				int tempint = 0;
		}
}

	return 0;
}

//∂®“Â∑¥¿°Õ¯¬Á^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
float feedback(Node_value &nva, Parameter &par, Inputlayer &inp,int epoch){
	Gradparameter gpr;
	Functions fun;
	float error = 0;
	float acc = 0;
	bool stocastic_flag = false;
	int count = 0;
	error /= output_layer_node_number;
	for (int i = 0; i< first_hiden_layer_node_number; i++)
	{
		nva.first_hiden_layer_node_grad_value[i] = 0;
		gpr.first_hiden_layer_bias_grad_array[i] = 0;
		for (int j = 0; j < second_hiden_layer_node_number; j++)
		{
			gpr.hh_weight_grad_array[i][j] = 0;
		}
	}
	for (int i = 0; i < second_hiden_layer_node_number; i++)
	{
		nva.second_hiden_layer_node_grad_value[i] = 0;
		gpr.second_hiden_layer_bias_grad_array[i] = 0;
		for (int j = 0; j < output_layer_node_number; j++)
		{
			gpr.ho_weight_grad_array[i][j] = 0;
		}
	}
	for (int i = 0; i < input_layer_node_number; i++)
	{
		for (int j = 0; j < first_hiden_layer_node_number; j++)
		{
			gpr.ih_weight_grad_array[i][j] = 0;
		}
	}
	for (int i = 0; i < output_layer_node_number; i++)
		gpr.output_layer_bias_grad_array[i] = 0;
	for (int n = 0; n < data_number; n++)
	{
		//Õ®π˝—≠ª∑«ÛΩ‚Ã›∂»≤¢∂‘»®÷ÿ∫Õ∆´÷√Ω¯––”≈ªØ**************

		int max_id = 0;
		float temp_max = 0;
		//«ÛΩ‚ ‰≥ˆ≤„µƒ∆´÷√Ã›∂»≤¢∂‘∆´÷√”≈ªØ∆´÷√£ª
		for (int i = 0; i<output_layer_node_number; i++) {
			if (i == inp.label[n])
			{
				gpr.output_layer_bias_grad_array[i] += nva.output_layer_node_value[n][i] - 1;
				error += (nva.output_layer_node_value[n][i] - 1)*(nva.output_layer_node_value[n][i] - 1) / output_layer_node_number;
			}
			else
			{
				gpr.output_layer_bias_grad_array[i] += nva.output_layer_node_value[n][i] - 0;
				error += (nva.output_layer_node_value[n][i] - 0)*(nva.output_layer_node_value[n][i] - 0) / output_layer_node_number;
			}


			if (temp_max < nva.output_layer_node_value[n][i])
			{
				temp_max = nva.output_layer_node_value[n][i];
				max_id = i;
			}
		}
		if (max_id == inp.label[n])
			acc = acc + 1;
		//«ÛΩ‚µ⁄∂˛“˛≤ÿ≤„µΩ ‰≥ˆ≤„»®÷ÿÃ›∂»≤¢”≈ªØ»®÷ÿ£ª
		for (int i = 0; i<second_hiden_layer_node_number; i++) {
			for (int j = 0; j<output_layer_node_number; j++) {
				if (j == inp.label[n])
					gpr.ho_weight_grad_array[i][j] += (nva.output_layer_node_value[n][j] - 1)*nva.second_hiden_layer_node_value[n][i];
				else
					gpr.ho_weight_grad_array[i][j] += (nva.output_layer_node_value[n][j] - 0)*nva.second_hiden_layer_node_value[n][i];
				if (fabs(gpr.ho_weight_grad_array[i][j])>100)
					int tempint = 0;
			}
		}
		
		//Õ®π˝—≠ª∑¿€º”¿¥∂®“Â ‰≥ˆ≤„ŒÛ≤Ó∫Ø ˝œÚµ⁄∂˛“˛≤ÿ≤„µƒ¥´µ›£ª
		for (int i = 0; i<second_hiden_layer_node_number; i++) {
			nva.second_hiden_layer_node_grad_value[i] = 0;
			for (int j = 0; j<output_layer_node_number; j++) {
				if (j == inp.label[n])
					nva.second_hiden_layer_node_grad_value[i] += (nva.output_layer_node_value[n][j] - 1)*par.ho_weight_array[i][j] * fun.gradsigmoid(nva.second_hiden_layer_node_value[n][i]);
				else
					nva.second_hiden_layer_node_grad_value[i] += (nva.output_layer_node_value[n][j] - 0)*par.ho_weight_array[i][j] * fun.gradsigmoid(nva.second_hiden_layer_node_value[n][i]);
			}
		}
		//«Ûµ⁄∂˛“˛≤ÿ≤„µƒÃ›∂»∆´÷√Ã›∂»≤¢”≈ªØ£ª
		for (int i = 0; i<second_hiden_layer_node_number; i++) {
			gpr.second_hiden_layer_bias_grad_array[i] += nva.second_hiden_layer_node_grad_value[i] ;
		}
		//«Ûµ⁄“ª“˛≤ÿ≤„µΩµ⁄∂˛“˛≤ÿ≤„»®÷ÿÃ›∂»≤¢”≈ªØ£ª
		for (int i = 0; i<first_hiden_layer_node_number; i++) {
			for (int j = 0; j<second_hiden_layer_node_number; j++) {
				gpr.hh_weight_grad_array[i][j] += nva.second_hiden_layer_node_grad_value[j] *nva.first_hiden_layer_node_value[n][i];
			}
		}
		
		//Õ®π˝—≠ª∑¿€º”¿¥∂®“Âµ⁄∂˛“˛≤ÿ≤„ŒÛ≤Ó∫Ø ˝œÚµ⁄“ª“˛≤ÿ≤„µƒ¥´µ›£ª
		for (int i = 0; i<first_hiden_layer_node_number; i++) {
			nva.first_hiden_layer_node_grad_value[i] = 0;
			for (int j = 0; j<second_hiden_layer_node_number; j++) {
				nva.first_hiden_layer_node_grad_value[i] += nva.second_hiden_layer_node_grad_value[j] * fun.gradsigmoid(nva.first_hiden_layer_node_value[n][j])*par.hh_weight_array[i][j];
			}
		}
		//«Û ‰»Î≤„µΩµ⁄“ª“˛≤ÿ≤„µƒ»®÷ÿÃ›∂»≤¢”≈ªØ£ª
		for (int i = 0; i<input_layer_node_number; i++) {
			for (int j = 0; j<first_hiden_layer_node_number; j++) {
				gpr.ih_weight_grad_array[i][j] += nva.first_hiden_layer_node_grad_value[j] *inp.input_data[n][i];
			}
		}
		//«Ûµ⁄“ª“˛≤ÿ≤„∆´÷√Ã›∂»≤¢”≈ªØ£ª
		for (int i = 0; i<first_hiden_layer_node_number; i++) {
			gpr.first_hiden_layer_bias_grad_array[i] = nva.first_hiden_layer_node_grad_value[i];
		}
	}
	if (count%batch_size == 0)
	{
		count = 0;;
		stocastic_flag = true;
	}
	count++;
	if (stocastic_flag)
	{
		for (int i = 0; i<second_hiden_layer_node_number; i++) {
			for (int j = 0; j<output_layer_node_number; j++) {
				par.ho_weight_array[i][j] -= learning_rate*gpr.ho_weight_grad_array[i][j] / batch_size;
			}
		}
		for (int i = 0; i<output_layer_node_number; i++) {
			par.output_layer_bias_array[i] -= learning_rate*gpr.output_layer_bias_grad_array[i] / batch_size;
		}
		for (int i = 0; i<second_hiden_layer_node_number; i++) {
			for (int j = 0; j<output_layer_node_number; j++) {

			}
			par.second_hiden_layer_bias_array[i] -= learning_rate*gpr.second_hiden_layer_bias_grad_array[i] / batch_size;
		}
		for (int i = 0; i<first_hiden_layer_node_number; i++) {
			for (int j = 0; j<second_hiden_layer_node_number; j++) {
				par.hh_weight_array[i][j] -= learning_rate*gpr.hh_weight_grad_array[i][j] / batch_size;
			}
		}
		for (int i = 0; i<input_layer_node_number; i++) {
			for (int j = 0; j<first_hiden_layer_node_number; j++) {
				par.ih_weight_array[i][j] -= learning_rate*gpr.ih_weight_grad_array[i][j] / batch_size;
			}
		}
		for (int i = 0; i<first_hiden_layer_node_number; i++) {
			par.first_hiden_layer_bias_array[i] -= learning_rate*gpr.first_hiden_layer_bias_grad_array[i] / batch_size;
		}
		for (int i = 0; i< first_hiden_layer_node_number; i++)
		{
			nva.first_hiden_layer_node_grad_value[i] = 0;
			gpr.first_hiden_layer_bias_grad_array[i] = 0;
			for (int j = 0; j < second_hiden_layer_node_number; j++)
			{
				gpr.hh_weight_grad_array[i][j] = 0;
			}
		}
		for (int i = 0; i < second_hiden_layer_node_number; i++)
		{
			nva.second_hiden_layer_node_grad_value[i] = 0;
			gpr.second_hiden_layer_bias_grad_array[i] = 0;
			for (int j = 0; j < output_layer_node_number; j++)
			{
				gpr.ho_weight_grad_array[i][j] = 0;
			}
		}
		for (int i = 0; i < input_layer_node_number; i++)
		{
			for (int j = 0; j < first_hiden_layer_node_number; j++)
			{
				gpr.ih_weight_grad_array[i][j] = 0;
			}
		}
		for (int i = 0; i < output_layer_node_number; i++)
			gpr.output_layer_bias_grad_array[i] = 0;
	}
	

	printf("epoch%d\t%f\t%f\n", epoch, error,acc/data_number);
	return 0;
}


/************************************************************************************
main function
************************************************************************************/
int main()
{
	Inputlayer inp;
	Parameter par;
	Node_value nva = Node_value();
	
	//∂¡»Î ˝æ›µΩinput ˝◊È£ª
	ifstream data_file;
	data_file.open("testdata.txt");
	for (int i = 0; i<data_number; i++)
	{
		for (int j = 0; j<data_dimension; j++)
		{
			data_file >> inp.input_data[i][j];
			//cout << inp.input_data[i][j];
		}
		//cout << endl;
	}
	data_file.close();
	//∂¡»ÎlabelµΩ ˝◊È
	ifstream label_file;
	label_file.open("datalabel.txt");
	for (int i = 0; i<data_number; i++) {
		label_file >> inp.label[i];
	}
	label_file.close();

	//∏¯»®÷ÿ°¢∆´÷√ ˝◊È∏≥»Œ“‚÷µ
	for (int i = 0; i<input_layer_node_number; i++) {//∏¯ ‰»Î≤„µΩµ⁄“ª“˛≤ÿ≤„»®÷ÿ ˝◊È∏≥÷µ£ª
		for (int j = 0; j<first_hiden_layer_node_number; j++) {
			par.ih_weight_array[i][j] = par.sample_from_gaussian(0, 0.01);
		}
	}
	for (int i = 0; i<first_hiden_layer_node_number; i++) {//∏¯µ⁄“ª“˛≤ÿ≤„µΩµ⁄∂˛“˛≤ÿ≤„»®÷ÿ ˝◊È∏≥÷µ£ª
		for (int j = 0; j<second_hiden_layer_node_number; j++) {
			par.hh_weight_array[i][j] = par.sample_from_gaussian(0, 0.01);
		}
	}
	for (int i = 0; i<second_hiden_layer_node_number; i++) {//∏¯µ⁄∂˛“˛≤ÿ≤„µΩ ‰≥ˆ≤„»®÷ÿ ˝◊È∏≥÷µ£ª
		for (int j = 0; j<output_layer_node_number; j++) {
			par.ho_weight_array[i][j] = par.sample_from_gaussian(0, 0.01);
		}
	}
	for (int i = 0; i<first_hiden_layer_node_number; i++) {
		par.first_hiden_layer_bias_array[i] = par.sample_from_gaussian(0, 0.01);
	}
	for (int i = 0; i<second_hiden_layer_node_number; i++) {
		par.second_hiden_layer_bias_array[i] = par.sample_from_gaussian(0, 0.01);
	}
	for (int i = 0; i<output_layer_node_number; i++) {
		par.output_layer_bias_array[i] = par.sample_from_gaussian(0, 0.01);
	}
	for (int i = 0; i < epoch_num; i++)
	{
		feedforward(inp, nva,par);
		feedback(nva, par,inp,i);
	}
	FILE* fp;
	fp=fopen("weights.txt", "w");
	for (int i = 0; i<input_layer_node_number; i++) {
		for (int j = 0; j<first_hiden_layer_node_number; j++) {
			fprintf(fp,"%f\t",par.ih_weight_array[i][j]);
		}
		fprintf(fp, "\n");
	}
	fprintf(fp, "\n");
	for (int i = 0; i < first_hiden_layer_node_number; i++)
		fprintf(fp, "%f\t", par.first_hiden_layer_bias_array[i]);
	fprintf(fp, "\n");
	fprintf(fp, "\n");
	for (int i = 0; i<first_hiden_layer_node_number; i++) {
		for (int j = 0; j<second_hiden_layer_node_number; j++) {
			fprintf(fp, "%f\t", par.hh_weight_array[i][j]);
		}
		fprintf(fp, "\n");
	}
	fprintf(fp, "\n");
	for (int i = 0; i < second_hiden_layer_node_number; i++)
		fprintf(fp, "%f\t", par.second_hiden_layer_bias_array[i]);
	fprintf(fp, "\n");
	fprintf(fp, "\n");
	for (int i = 0; i<second_hiden_layer_node_number; i++) {
		for (int j = 0; j<output_layer_node_number; j++) {
			fprintf(fp, "%f\t", par.ho_weight_array[i][j]);
		}
		fprintf(fp, "\n");
	}
	for (int i = 0; i < output_layer_node_number; i++)
		fprintf(fp, "%f\t", par.output_layer_bias_array[i]);
	fprintf(fp, "\n");
	fprintf(fp, "\n");
	fclose(fp);
	return 0;
}



