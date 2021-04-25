#ifndef DATA_OP_H
#define DATA_OP_H

/** \file
 *
 * Defines analyses to extract the functions called a function.
 */

#include <map>
#include <string>
#include <set>
#include "Halide.h"
#include <vector>
namespace Halide {
namespace Internal {
    enum class DataTransformMethod{
    REORDER,
    INTERLEAVE,
    SPLITY
};
namespace {

//DataOP is not used but help to remember user write an OP with function operator() and name()
class DataOP
{
public:
    // operate the vector<Expr>
    virtual void operator()(std::vector<Expr> &args){}
    // this operator for new_func(v1,v2,v3...) = old_func(v1*a+v3,v2...)
    // so the rargs and the largs are both differen with the origin args
    virtual void operator()(Func &lfunc,Func &rfunc){}
    // create a name for the new function = old_function+name();
    virtual std::string name(){return "$NULL";}
};

class ReorderOP: DataOP
{
public:
    // reorder means swap var:0 and 1;
    // func(x,y,z) = input_a(k,y,z)*input_b(x,k,z)->Br(k,x,z)
    virtual void operator()(std::vector<Expr> &args)
    {
        if (args.size()>1)
            std::swap(args[0],args[1]);
        else{
            //TODO throw an error or do nothing?
        }
    }
    // new_func(x,y,z) = old_func(y,x,z)
    virtual void operator()(Func &lfunc,Func &rfunc){
        int args_size = rfunc.args().size();
        std::vector<Var> lvars;
        for (int i=0;i<args_size;i++){
             std::string varname = lfunc.name()+"_"+ std::to_string(i);
             lvars.emplace_back(Var(varname));
        }
        //std::vector<Var> lvars = rvars;
        std::vector<Expr> rargs(args_size);
        for (size_t i = 0; i < rargs.size(); i++) {
            rargs[i] = lvars[i];
        }
        if (rargs.size()>1)
            std::swap(rargs[0],rargs[1]);
        else{
            //TODO throw an error or do nothing?
        }
        lfunc(lvars) = rfunc(rargs);
    }
    virtual std::string name(){return "$Reorder";}
};

class InterleaveOP: DataOP
{
private:
    int split_num_;
public:
    InterleaveOP(){split_num_=8;}
    InterleaveOP(int split_num):split_num_(split_num){}
    //func(x,y,b)=input_a(k,y,b)*input_b(x,k,b)->Br(x%8,k,x/8,b)
    //func(x,y,z) = ....OP new_func(x%a,y,x/a,z)  <--- old_func(x,y,z);
    virtual void operator()(std::vector<Expr> &args)
    {
        args.emplace_back(Expr());
        if (args.size()<3)
        {
            //TODO
        }else
        {
            
            for (size_t i=args.size()-1;i>1;i--)
            {
                args[i] = args[i-1];
            }
            args[2] = args[0]/split_num_;
            args[0] = args[0]%split_num_;
        }
    }
    //new_func(x,y,xo,z) = old_func(xo*a+x,y,z)
    virtual void operator()(Func &lfunc,Func &rfunc){
        int args_size = rfunc.args().size()+1;

        if (args_size<2){
            //TODO error?
            return;
        }
        std::vector<Var> lvars;
        for (int i=0;i<args_size;i++){
             std::string varname = lfunc.name()+"_"+ std::to_string(i);
             lvars.emplace_back(Var(varname));
        }
        //std::vector<Var> lvars = rvars;
        std::vector<Expr> rargs(args_size-1);
        for (size_t i = 0; i < rargs.size(); i++) {
            if (i==0)
                rargs[0] = lvars[0]*split_num_+lvars[2];
            else if (i==1)
                rargs[1] = lvars[1];
            else
                rargs[i] = lvars[i+1];
        }
        lfunc(lvars) = rfunc(rargs);
    }
    virtual std::string name(){return "$Interleave";}
};
class SplitYOP: DataOP
{
private:
    int split_num_;
public:
    SplitYOP(){split_num_=8;}
    SplitYOP(int split_num):split_num_(split_num){}
    //func(x,y)=input_a(x,y)->Br(x,y%16,y/16)
    virtual void operator()(std::vector<Expr> &args)
    {
        args.emplace_back(Expr());
        if (args.size()<2)
        {
            //TODO
        }else
        {
            
            for (size_t i=args.size()-1;i>1;i--)
            {
                args[i] = args[i-1];
            }
            args[2] = args[1]/split_num_;
            args[1] = args[1]%split_num_;
        }
    }
    //new_func(x,y,yo) = old_func(x,yo*16+y)
    virtual void operator()(Func &lfunc,Func &rfunc){
        int args_size = rfunc.args().size()+1;

        if (args_size<2){
            //TODO error?
            return;
        }
        std::vector<Var> lvars;
        for (int i=0;i<args_size;i++){
             std::string varname = lfunc.name()+"_"+ std::to_string(i);
             lvars.emplace_back(Var(varname));
        }
        //std::vector<Var> lvars = rvars;
        std::vector<Expr> rargs(args_size-1);
        for (size_t i = 0; i < rargs.size(); i++) {
            if (i==0)
                rargs[0] = lvars[0];
            else if (i==1)
                rargs[1] = lvars[1]*split_num_+lvars[2];
            else
                rargs[i] = lvars[i+1];
        }
        lfunc(lvars) = rfunc(rargs);
    }
    virtual std::string name(){return "$SplitY";}
};
//Im2ColOP:input(x+r0,y+r1,r2,n)->im2col(k,x,y,n) k=r0*r1*r0

// class Im2ColOP: DataOP
// {
// public:
//     Im2ColOP(){}
//     //SplitYOP(int split_num):split_num_(split_num){}
//     //func(x,y,co,n)+=input(x+r0,y+r1,r2,n)*kernel(r0,r1,r2,n)
//     // input(x+r0,y+r1,r2,n)->im2col(r0+r1*K+r2*K*K,x+y*W,n)
//     // im2col(v1,v2,v3) = input(v2%W+v1%K,v2/W+v1%(K*K)/K,v1/(K*K),v3)
//     // kernel_col(v1,n) = kernel(v1%K,v1%(K*K)/K,v1/(K*K),n)
//     //func(x,y,co,n)+=im2col(k,x,y,n)*kernel_col(k,n)
//     //input(x+r0,y+r1,r2,n)->im2col(k,x,y,n) k=r0*r1*r2
//     virtual void operator()(std::vector<Expr> &args)
//     {
//         args.emplace_back(Expr());
//         if (args.size()<2)
//         {
//             //TODO
//         }else
//         {
            
//             for (size_t i=args.size()-1;i>1;i--)
//             {
//                 args[i] = args[i-1];
//             }
//             args[2] = args[1]/split_num_;
//             args[1] = args[1]%split_num_;
//         }
//     }
//     //new_func(x,y,yo) = old_func(x,yo*16+y)
//     virtual void operator()(Func &lfunc,Func &rfunc){
//         int args_size = rfunc.args().size()+1;

//         if (args_size<2){
//             //TODO error?
//             return;
//         }
//         std::vector<Var> lvars;
//         for (int i=0;i<args_size;i++){
//              std::string varname = lfunc.name()+"_"+ std::to_string(i);
//              lvars.emplace_back(Var(varname));
//         }
//         //std::vector<Var> lvars = rvars;
//         std::vector<Expr> rargs(args_size-1);
//         for (size_t i = 0; i < rargs.size(); i++) {
//             if (i==0)
//                 rargs[0] = lvars[0];
//             else if (i==1)
//                 rargs[1] = lvars[1]*split_num_+lvars[2];
//             else
//                 rargs[i] = lvars[i+1];
//         }
//         lfunc(lvars) = rfunc(rargs);
//     }
//     virtual std::string name(){return "#SplitY";}
// };
}  // namespace unname
}  // namespace Internal
}  // namespace Halide

#endif
