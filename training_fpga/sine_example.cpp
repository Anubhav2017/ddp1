#include<iostream>
#include<math.h>
using namespace std;
void forward_fcc(float* x, float* w, float* y, float* b, int xdim, int ydim){

    for(int i=0; i< ydim;i++){
        y[i]= b[i];

        for (int j=0; j<xdim;j++){
            y[i]+= w[i*xdim+j]*x[j];
        }
    }

}

void backward_fcc(float* x, float* w, float* y, float* b, float* dx, float* dy, float* db, float* dw, int xdim, int ydim){
    //compute gradient of activations
    for(int i=0;i<xdim;i++){
        for(int j=0;j<ydim;j++){
            dx[i] = dy[j] * w[i+j*xdim];
        }
        
    }
    //compute gradient of weights
    for(int i=0;i<ydim;i++){
        for(int j=0;j<xdim;j++){
            dw[i*xdim+j] = dy[i]*x[j];
        }
    }

    //compute gradient of biases
    for (int i=0;i<ydim;i++){
        db[i] = dy[i];
    }
}

void forward_softmax(float* z, float* a, int size_t){

    float* expz = new float[size_t];

    for(int i=0; i< size_t;i++){
        expz[i] = exp(z[i]);
    }
    float expsum = 0;
    for (int i=0; i<size_t;i++){
        expsum += expz[i];
    }
    for(int i=0;i<size_t;i++){
        z[i]= expz[i]/expsum;
    }
    delete[] expz;

}

float mse_loss(float* pred, float* truth, int dim){
    float loss=0;
    for(int i=0;i<dim;i++){
        loss+= pow(pred[i]-truth[i],2);
    }
    loss = loss/dim;
    return loss;
}

void mse_gradient(float* pred, float* truth, float* grad,int dim){

    for(int i=0;i<dim;i++){
        grad[i]=(pred[i]-truth[i])/dim;
    }

}

void cross_entropy_derivative(float* q,int label,float* dz, int dim){
    for(int i=0;i<dim;i++){
        dz[i] = q[i];
    }
    dz[label] -= 1; 
}

void create_sine_data(float* input, float* output,int size){

    for (int i=0;i<size;i++){
        input[i] = (float) rand()/RAND_MAX;
    }
    for(int i=0;i<size;i++){
        output[i] = sin(input[i])+0.1*(float) rand()/RAND_MAX;
    }

}

void init_weights_random(float* w, int size_t){
    for (int i=0;i<size_t;i++){
        w[i]=0.1*(float) rand()/RAND_MAX;
    }
}

void train_model(){

    float train_x[1000]={};
    float train_y[1000]={};
    create_sine_data(train_x,train_y,1000);

    float w1[8]={};
    float w2[8]={};
    float b1[8]={};
    float b2[1]={};

    init_weights_random(w1,8);
    init_weights_random(w2,8);
    init_weights_random(b1,8);
    init_weights_random(b2,1);

}


int main(){
    float x[1000] = {1.0, 2.0};

    float w[2] = {0.5, 0.1};
    float b[1]= {0.01};
    float z[1]={0};

    float dx[2] = {1.0, 2.0};
    float dw[2] = {0.5, 0.1};
    float db[1] = {0.01};
    float dz[1] = {0.2};

    forward_fcc(x,w,z,b,2,1);
    backward_fcc(x,w,z,b,dx,dz,db,dw,2,1);

    cout << "dz=" << dz[0] <<"\n";
    cout << "db=" << db[0] <<"\n";
    cout << "dx = " << dx[0] <<" "<< dx[1] << "\n";
    cout << "dw = " << dw[0] << " " <<dw[1] << "\n";

    return 0;
}