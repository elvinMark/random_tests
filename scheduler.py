import bisect 
import math

def create_lr_scheduler(args):
    if args.sched == "constant":
        def learning_rate(init,epoch):
            return init
        return learning_rate
    
    elif args.sched == "step":        
        def learning_rate(init, epoch):
            return init * args.gamma ** (epoch // args.step_size)
        return learning_rate
    
    elif args.sched == "multistep":
        def learning_rate(init,epoch):
            idx = bisect.bisect_left(args.milestones,epoch)
            return init * args.gamma ** idx
        return learning_rate
    
    elif args.sched == "cosine":
        def learning_rate(init,epoch):
            return args.eta_min + 0.5 * (args.lr  - args.eta_min)*(1  + math.cos(epoch * math.pi / args.T_max))
        return learning_rate
    
    else:
        return None


def create_warmup(sched,args):
    def linear_scheduler(epoch):
        return args.lower_lr + (args.lr - args.lower_lr) * epoch / args.warmup
    def warmup_scheduler(init,epoch):
        if epoch <= args.warmup:
            return linear_scheduler(epoch)
        else:
            return sched(init,epoch)

    return warmup_scheduler
            
