#ifndef BOUND_ESTIMATE_H
#define BOUND_ESTIMATE_H

/** \file
 *
 * Defines analyses to extract the functions called a function.
 */

#include <map>
#include <string>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include "Halide.h"
namespace Halide {
namespace Internal {

using std::map;
using std::pair;
using std::string;

namespace {
/* Find all the internal halide calls in an expr */

// 找到一个rvalue包含target_name的definition。
class FindDefinition: public IRVisitor {
private:
    bool success;
public:
    std::string target_name;
    using IRVisitor::visit;
    FindDefinition(){
        success = false;
    }
    FindDefinition(std::string target){
        target_name = target;
        success = false;
    }
    bool find_target() {
        return success;
    }

    void visit(const Call *call) override {
        IRVisitor::visit(call);
        if (call->call_type == Call::Halide && call->func.defined()) {
            Function f(call->func);
            if (f.name()==target_name||f.name().find(target_name+"$")!=string::npos)
                success = true;
        }
    }
};
class SetBounds: public IRVisitor {
private:
    std::map<std::string,std::pair<int,int>> *bounds_;
    std::unordered_map<std::string,int> vartoidx_;
    std::string consumer_;
    std::string producer_;
    int var_idx_;
    std::pair<int,int> res;
public:
    using IRVisitor::visit;
    SetBounds() = default;
    SetBounds(std::string consumer,std::string producer,int var_idx,std::unordered_map<std::string,int> vartoidx, std::map<std::string,std::pair<int,int>> *b){
        bounds_ = b;
        consumer_ = consumer;
        producer_ = producer;
        var_idx_ = var_idx;
        vartoidx_ = vartoidx;
        res = std::make_pair(0,0);
    }
    std::pair<int,int> get_bounds() {
        return res;
    }
    void visit(const Variable *node) override {
        std::string name = node->name;
        //std::cout<<"var name:"<<name<<std::endl;
        if (node->reduction_domain.defined())
        {
            auto domains = node->reduction_domain.domain();
            //std::cout<<"domain size:"<<domains.size()<<std::endl;
            for (auto d:domains)
            {
                //std::cout<<"dname:"<<d.var<<std::endl;
                if (name==d.var)
                {
                    //std::cout<<"to int"<<std::endl;
                    int i_min = *as_const_int(d.min);
                    
                    int i_extent = *as_const_int(d.extent);
                    //std::cout<<"end extent int"<<std::endl;
                    res.first = i_min;
                    res.second = i_min+i_extent-1;
                    (*bounds_)[name] = res;
                    //std::cout<<"producer:"<<producer_<<"consumer:"<<consumer_<<"name:"<<name<<"res:"<<i_min<<" "<<res.second<<std::endl;
                    return;
                }
                
            }
            std::cout<<" a RDom variable but not find the bound"<<std::endl;
            return;
        }
        if (vartoidx_.find(name)==vartoidx_.end())
        {
            //TODO
            std::cout<<"error bound estimate:var to idx fail:"<<name<<std::endl;
        }
        int idx = vartoidx_[name];
        std::string bound_name = consumer_+"."+std::to_string(idx);
        if ((*bounds_).find(bound_name)==(*bounds_).end())
        {
            //TODO
            std::cout<<"error bound estimate:wrong variable name:"<<bound_name<<std::endl;
        }
        //std::cout<<"find bound name:"<<bound_name<<" range:"<<(*bounds_)[bound_name].first<<" ---- "<<(*bounds_)[bound_name].second<<std::endl;
        res = (*bounds_)[bound_name];
        //std::cout<<"var name:"<<name<<std::endl;
    }
    void visit(const IntImm *node) override {
        int value = node->value;
        res = std::make_pair(value,value);
    }
    void visit(const Add *node) override {
        //IRVisitor::visit(node);
        std::pair<int,int> bound_a;
        std::pair<int,int> bound_b;
        SetBounds setbounds_a(consumer_,producer_,var_idx_,vartoidx_,bounds_);
        SetBounds setbounds_b(consumer_,producer_,var_idx_,vartoidx_,bounds_);
        node->a.accept(&setbounds_a);
        node->b.accept(&setbounds_b);
        bound_a = setbounds_a.get_bounds();
        bound_b = setbounds_b.get_bounds();
        res.first = bound_a.first+bound_b.first;
        res.second = bound_a.second+bound_b.second;
    }
    void visit(const Sub *node) override {
        //IRVisitor::visit(node);
        std::pair<int,int> bound_a;
        std::pair<int,int> bound_b;
        SetBounds setbounds_a(consumer_,producer_,var_idx_,vartoidx_,bounds_);
        SetBounds setbounds_b(consumer_,producer_,var_idx_,vartoidx_,bounds_);
        node->a.accept(&setbounds_a);
        node->b.accept(&setbounds_b);
        bound_a = setbounds_a.get_bounds();
        bound_b = setbounds_b.get_bounds();
        res.first = bound_a.first-bound_b.second;
        res.second = bound_a.second-bound_b.first;
    }
    void visit(const Mul *node) override {
        std::pair<int,int> bound_a;
        std::pair<int,int> bound_b;
        SetBounds setbounds_a(consumer_,producer_,var_idx_,vartoidx_,bounds_);
        SetBounds setbounds_b(consumer_,producer_,var_idx_,vartoidx_,bounds_);
        node->a.accept(&setbounds_a);
        node->b.accept(&setbounds_b);
        bound_a = setbounds_a.get_bounds();
        bound_b = setbounds_b.get_bounds();
        res.first = bound_a.first*bound_b.first;
        res.second = bound_a.second*bound_b.second;
    }
    void visit(const Div *node) override {
        std::pair<int,int> bound_a;
        std::pair<int,int> bound_b;
        SetBounds setbounds_a(consumer_,producer_,var_idx_,vartoidx_,bounds_);
        SetBounds setbounds_b(consumer_,producer_,var_idx_,vartoidx_,bounds_);
        node->a.accept(&setbounds_a);
        node->b.accept(&setbounds_b);
        bound_a = setbounds_a.get_bounds();
        bound_b = setbounds_b.get_bounds();
        res.first = bound_a.first/bound_b.second;
        if (bound_b.first!=0)
            res.second = bound_a.second/bound_b.first;
        else{
            //TODO 
            std::cout<<"error bound estimate: div 0!"<<std::endl;
        }
    }
    void visit(const Mod *node) override {
        std::pair<int,int> bound_a;
        std::pair<int,int> bound_b;
        SetBounds setbounds_a(consumer_,producer_,var_idx_,vartoidx_,bounds_);
        SetBounds setbounds_b(consumer_,producer_,var_idx_,vartoidx_,bounds_);
        node->a.accept(&setbounds_a);
        node->b.accept(&setbounds_b);
        bound_a = setbounds_a.get_bounds();
        bound_b = setbounds_b.get_bounds();
        res.first = bound_a.first%bound_b.second;
        if (bound_a.second>bound_b.second)
        {
            res.second = bound_b.second-1;
        }else
        {
            res.second = bound_a.second;
        }
    }
};
class FindBounds: public IRVisitor {
private:
    std::map<std::string,std::pair<int,int>> *bounds_;
    std::string function_name_;
    std::unordered_map<std::string,int> vartoidx_;
public:
    using IRVisitor::visit;
    FindBounds(){
        bounds_ = nullptr;
    }
    FindBounds(std::string function_name,std::unordered_map<std::string,int> vartoidx,std::map<std::string,std::pair<int,int>> *b){
        bounds_ = b;
        function_name_ = function_name;
        vartoidx_= vartoidx;
    }
    // bool find_target() {
    //     return success;
    // }

    void visit(const Call *call) override {
        IRVisitor::visit(call);
        if (call->call_type == Call::Halide && call->func.defined()) {
            Function f(call->func);
            auto expr = call->args;
            //std::cout<<"call name:"<<f.name()<<std::endl;
            //auto expr = f.args();
            for (unsigned int i=0;i<expr.size();i++)
            {

                SetBounds setbounds(function_name_,f.name(),i, vartoidx_, bounds_);
                //std::cout<<"accept:"<<i<<std::endl;
                expr[i].accept(&setbounds);
                //std::cout<<"end accept:"<<i<<std::endl;
                std::pair<int,int> bound = setbounds.get_bounds();
                std::string bound_name = f.name()+"."+std::to_string(i);
                if ((*bounds_).find(bound_name)!=(*bounds_).end())
                {
                    (*bounds_)[bound_name].first = std::min((*bounds_)[bound_name].first,bound.first);
                    (*bounds_)[bound_name].second = std::max((*bounds_)[bound_name].second,bound.second);
                }else{
                    (*bounds_)[bound_name] = bound;
                    //std::cout<<"set bound:"<<bound_name<<"  "<<bound.first<<" ---->"<<bound.second<<std::endl;
                }
                
            }
        }
    }
};
class FindLoopOrder: public IRVisitor {
private:
    //std::map<std::string,std::pair<int,int>> *bounds_;
    std::vector<std::unordered_set<std::string>> orders_;
    std::unordered_map<std::string,int> vartoidx_;
    int idx_;
public:
    using IRVisitor::visit;
    FindLoopOrder(){
        idx_=0;
    }
    FindLoopOrder(std::unordered_map<std::string,int> &vartoidx,int size){
        vartoidx_ = vartoidx;
        idx_=0;
        orders_.resize(size);
    }
    std::vector<std::unordered_set<std::string>> get_loop_order()
    {
        return orders_;
    }
    void set_order(int idx){
        idx_=idx;
    }
    void visit(const Variable *node) override {
        std::string name = node->name;
        if (vartoidx_.find(name)!=vartoidx_.end())
            orders_[idx_].insert(name);
    }
};
class FindOrder: public IRVisitor {
private:
    //std::map<std::string,std::pair<int,int>> *bounds_;
    std::string target_name_;
    std::unordered_map<std::string,int> vartoidx_;
    std::vector<std::unordered_set<std::string>> orders_;
    bool misorder_;
public:
    using IRVisitor::visit;
    FindOrder(){
        //bounds_ = nullptr;
        misorder_ = false;
    }
    FindOrder(std::string target_name,std::unordered_map<std::string,int> vartoidx){
        //bounds_ = b;
        target_name_ = target_name;
        vartoidx_ = vartoidx;
        misorder_ = false;
    }
    std::vector<std::unordered_set<std::string>> get_orders()
    {
        return orders_;
    }
    std::string get_target(){
        return target_name_;
    }
    bool IsMisorder()
    {
        return misorder_;
    }
    void visit(const Call *call) override {
        IRVisitor::visit(call);
        if (call->call_type == Call::Halide && call->func.defined()) {
            Function f(call->func);
            FindLoopOrder find_loop_order(vartoidx_,call->args.size());
            if (f.name() == target_name_||f.name().find(target_name_+"$")!=string::npos)
            {
                if (f.name().find(target_name_+"#")!=string::npos)
                {
                    target_name_ = f.name();
                }
                auto exprs = call->args;
                for (unsigned int i = 0;i<exprs.size();i++)
                {
                    auto expr = exprs[i];
                    find_loop_order.set_order(i);
                    expr.accept(&find_loop_order);
                }
                orders_ = find_loop_order.get_loop_order();
                if (orders_.size()>1)
                {
                    bool first2second=false;
                    bool second2first=false;
                    for (auto iter=orders_[0].begin();iter!=orders_[0].end();iter++)
                    {
                        std::string var_name = (*iter);
                        if (vartoidx_.find(var_name)==vartoidx_.end())
                        {
                            //TODO
                            std::cout<<"error var name:"<<var_name<<std::endl;
                            continue;
                        }
                            
                        if (vartoidx_[var_name]==1)
                        {
                            first2second=true;
                            break;
                        }
                    }
                    for (auto iter=orders_[1].begin();iter!=orders_[1].end();iter++)
                    {
                        std::string var_name = (*iter);
                        if (vartoidx_.find(var_name)==vartoidx_.end())
                        {
                            //TODO
                            std::cout<<"error var name:"<<var_name<<std::endl;
                            continue;
                        }
                        if (vartoidx_[var_name]==0)
                        {
                            second2first=true;
                            break;
                        }
                    }
                    if (second2first&&first2second)
                    {
                        misorder_=true;
                    }
                }
            }
        }
    }
};

}  // namespace

std::vector<Function> find_input_function(const std::vector<Function> &outputs)
{
    std::unordered_set<string> visited_function; 
    std::queue<Function> env_queue;
    for (const Function &output:outputs)
    {
        visited_function.insert(output.name());
        env_queue.push(output);
    }
    std::vector<Function> inputs;
    while (!env_queue.empty())
    {
        Function func = env_queue.front();
        env_queue.pop();
        map<string, Function> env = find_direct_calls(func);
        if (env.size()==0)
            inputs.emplace_back(func);
        for (auto iter=env.begin();iter!=env.end();iter++)
        {
            if (visited_function.find(iter->first)==visited_function.end())
            {
                visited_function.insert(iter->first);
                env_queue.push(iter->second);
            }
                
        }
    }
    return inputs;
}
std::vector<Definition> find_target_definition(const Function &func,const std::string &target)
{
    std::vector<Definition> definitions;
    //definitions.emplace_back(func.definition());
    FindDefinition find_definition(target);
    func.definition().accept(&find_definition);
    if (find_definition.find_target())
    {
        definitions.emplace_back(func.definition());
    }
    std::vector<Definition> updates = func.updates();
    for (Definition definition:updates)
    {
        FindDefinition find_update(target);
        definition.accept(&find_update);
        if (find_update.find_target())
        {
            definitions.emplace_back(definition);
        }
    }
    return definitions;
}
std::vector<std::pair<Definition,std::string>> find_definition(const  std::vector<Function> &outputs,const std::string &target)
{
    //返回一个涉及target的definition
    std::unordered_set<string> visited_function; 
    std::queue<Function> env_queue;
    for (const  Function &output:outputs)
    {
        visited_function.insert(output.name());
        env_queue.push(output);
    }
    std::vector<std::pair<Definition,std::string>> res;
    while (!env_queue.empty())
    {
        Function func = env_queue.front();
        env_queue.pop();
        map<string, Function> env = find_direct_calls(func);
        std::vector<Definition> definitions = find_target_definition(func,target);
        if (definitions.size()>0)
        {
            for (Definition definition:definitions)
            {

                res.emplace_back(make_pair(definition,func.name()));
            }
                
        }
        for (auto iter=env.begin();iter!=env.end();iter++)
        {
            if (visited_function.find(iter->first)==visited_function.end())
            {
                visited_function.insert(iter->first);
                env_queue.push(iter->second);
            }
                
        }
    }
    return res;
}
void update_bounds(std::string function_name,const Definition& def, std::map<std::string,std::pair<int,int> > &bounds)
{
    std::vector<Expr> lvalue = def.args();
    std::vector<Expr> rvalue = def.values();
    //std::cout<<"def args size:"<<lvalue.size()<<std::endl;
    //std::cout<<"def values size:"<<rvalue.size()<<std::endl;
    std::unordered_map<std::string,int> args_to_idx;
    for (unsigned int i=0;i<lvalue.size();i++)
    {
        if (const Halide::Internal::Variable *v= lvalue[i].as<Halide::Internal::Variable>())
        {
            args_to_idx[v->name] = i;
        } 
    }
    //std::string function_name,std::unordered_map<std::string,int> vartoidx,std::map<std::string,std::pair<int,int>> *b
    FindBounds findbound(function_name,args_to_idx,&bounds);
    //std::cout<<"estimage function bound:"<<function_name<<std::endl;
    for (auto expr:rvalue)
    {
        expr.accept(&findbound);
    }
    
    return;
}
void propagate_bounds(Function &f, std::map<std::string,std::pair<int,int> > &bounds)
{
    //std::cout<<"good"<<std::endl;
    std::vector<Bound> bound = f.schedule().estimates();
    for (unsigned int i=0;i<bound.size();i++)
    {
        Bound b = bound[i];
        std::string bound_name = f.name()+"."+ std::to_string(i);
        int i_min = *as_const_int(b.min);
        int i_extent = *as_const_int(b.extent);
        if (bounds.find(bound_name)==bounds.end())
        {
            bounds[bound_name] = std::make_pair(i_min,i_min+i_extent-1); 
        }
        
        //std::cout<<bound_name<<" "<<i_min<<" ---- "<<i_min+i_extent-1<<std::endl;
    }
    Definition def = f.definition();
    //std::cout<<"init"<<std::endl;
    update_bounds(f.name(),def,bounds);
    //std::cout<<"updatesssss:"<<f.name()<<std::endl;
    std::vector<Definition> updates = f.updates();
    for (Definition def:updates){
        //std::cout<<"updates:"<<f.name()<<std::endl;
        update_bounds(f.name(),def,bounds);
    }
}
// function..definition().schedule().rvars()可以获得RDom

void estimate_bound(std::vector<Function> &outputs,std::map<std::string,std::pair<int,int> >& bounds)
{
    //std::cout<<"estimate first"<<std::endl;
    map<string, Function> env;
    for (Function f : outputs) {
        populate_environment(f, env);
    }
    std::vector<std::string> order = topological_order(outputs, env);
    for (int i=order.size()-1;i>=0;i--)
    {
        Function f = env[order[i]];
        propagate_bounds(f,bounds);
    }
}
}  // namespace Internal
}  // namespace Halide

#endif
