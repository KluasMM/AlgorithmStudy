package com.mt.algorithm.algorithmstudy.basic;

import lombok.AllArgsConstructor;
import lombok.NoArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.Arrays;
import java.util.List;

/**
 * @Description
 * @Author T
 * @Date 2022/9/1
 */
@Service
public class PartNineDp {

    /**
     * Definition for a binary tree node.
     */
    @AllArgsConstructor
    @NoArgsConstructor
    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int val) {
            this.val = val;
        }
    }

    /**
     * 509. 斐波那契数
     * <p>
     * 斐波那契数 （通常用 F(n) 表示）形成的序列称为 斐波那契数列 。
     * 该数列由 0 和 1 开始，后面的每一项数字都是前面两项数字的和。也就是：
     * F(0) = 0，F(1) = 1
     * F(n) = F(n - 1) + F(n - 2)，其中 n > 1
     * 给定 n ，请计算 F(n) 。
     *
     * @param n
     * @return
     */
    public int leetCode509(int n) {
        if (n < 2) return n;

        int result = 0;
        int lastOne = 1;
        int lastTwo = 0;
        for (int i = 2; i <= n; i++) {
            result = lastOne + lastTwo;
            lastTwo = lastOne;
            lastOne = result;
        }

        return result;
    }

    /**
     * 322. 零钱兑换
     * <p>
     * 给你一个整数数组 coins ，表示不同面额的硬币；以及一个整数 amount ，表示总金额。
     * <p>
     * 计算并返回可以凑成总金额所需的 最少的硬币个数 。如果没有任何一种硬币组合能组成总金额，返回 -1 。
     * <p>
     * 你可以认为每种硬币的数量是无限的。
     *
     * @param coins
     * @param amount
     * @return
     */
    int[] amountStore322;

    public int leetCode322(int[] coins, int amount) {
        /*
         * 自底向上
         */
        int[] dp = new int[amount + 1];
        Arrays.fill(dp, amount + 1);

        //base case
        dp[0] = 0;
        for (int i = 0; i < dp.length; i++) {
            for (int coin : coins) {
                if (i - coin < 0) {
                    System.out.printf("当前amount:%s,coin：%s 没有结果跳过 %n", i, coin);
                    continue;
                }
                dp[i] = Math.min(dp[i], dp[i - coin] + 1);
                System.out.printf("当前amount:%s,coin：%s, 计算结果：%s %n", i, coin, dp[i]);
            }
        }

        return dp[amount] == amount + 1 ? -1 : dp[amount];

        /*
         * 自顶向下
         */
        /*amountStore322 = new int[amount + 1];
        Arrays.fill(amountStore322, -2);
        return dp322(coins, amount);*/
    }

    private int dp322(int[] coins, int amount) {
        //base case
        if (amount == 0) return 0;
        if (amount < 0) return -1;

        //has result
        if (amountStore322[amount] != -2) return amountStore322[amount];
        System.out.printf("当前amount:%s,没有结果%n", amount);

        int result = Integer.MAX_VALUE;

        for (int coin : coins) {
            int temp = dp322(coins, amount - coin);
            if (temp == -1) continue;
            result = Math.min(result, temp + 1);
            System.out.printf("当前amount:%s,coin：%s, 计算结果：%s %n", amount, coin, result);
        }

        amountStore322[amount] = result == Integer.MAX_VALUE ? -1 : result;
        System.out.printf("当前amount:%s,最终计算结果：%s %n", amount, amountStore322[amount]);
        return amountStore322[amount];
    }

    /**
     * 300. 最长递增子序列
     * <p>
     * 给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。
     * <p>
     * 子序列 是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。
     * 例如，[3,6,2,7] 是数组 [0,3,1,6,2,2,7] 的子序列。
     *
     * @param nums
     * @return
     */
    public int leetCode300(int[] nums) {
        /*
         * 解题思路1：动态规划
         *  dp数组：代表以当前元素结尾的最大递增子序列
         *  转换方程：找到i之前所有比nums[i]数值小的对应的dp结果 取最大的dp结果+1
         *  最后结果就是dp的最大值
         */
        int result = 1;
        int[] dp = new int[nums.length];
        //全部填充为1 因为单个元素的最大子序列为1 即元素自己 后续遍历可以逐一初始化 替代这一步
        // Arrays.fill(dp, 1);

        //base case
        dp[0] = 1;

        //先计算出nums对应的dp数据
        for (int i = 1; i < nums.length; i++) {
            //初始化dp 省去Arrays.fill(dp, 1);
            dp[i] = 1;
            for (int j = 0; j < i; j++) {
                if (nums[j] < nums[i]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
            //省去后面从dp中找最大值
            result = Math.max(result, dp[i]);
        }

        //dp数组中最大的值就是结果
        // for (int res : dp) {
        //     result = Math.max(result, res);
        // }

        return result;

        /*
         * 解题思路2：二分法 动态规划
         *  看看就行了
         *  leetCode354:俄罗斯套娃 时需要用到
         */
        /*return lengthOfLISSolveByBinary(nums);*/
    }

    private int lengthOfLISSolveByBinary(int[] nums) {
        int[] top = new int[nums.length];
        // 牌堆数初始化为 0
        int piles = 0;
        for (int i = 0; i < nums.length; i++) {
            // 要处理的扑克牌
            int poker = nums[i];

            /***** 搜索左侧边界的二分查找 *****/
            int left = 0, right = piles;
            while (left < right) {
                int mid = (left + right) / 2;
                if (top[mid] > poker) {
                    right = mid;
                } else if (top[mid] < poker) {
                    left = mid + 1;
                } else {
                    right = mid;
                }
            }
            /*********************************/

            // 没找到合适的牌堆，新建一堆
            if (left == piles) piles++;
            // 把这张牌放到牌堆顶
            top[left] = poker;
        }
        // 牌堆数就是 LIS 长度
        return piles;
    }

    /**
     * 354. 俄罗斯套娃信封问题
     * <p>
     * 给你一个二维整数数组 envelopes ，其中 envelopes[i] = [wi, hi] ，表示第 i 个信封的宽度和高度。
     * <p>
     * 当另一个信封的宽度和高度都比这个信封大的时候，这个信封就可以放进另一个信封里，如同俄罗斯套娃一样。
     * <p>
     * 请计算 最多能有多少个 信封能组成一组“俄罗斯套娃”信封（即可以把一个信封放到另一个信封里面）。
     * <p>
     * 注意：不允许旋转信封。
     *
     * @param envelopes
     * @return
     */
    public int leetCode354(int[][] envelopes) {
        /*
         * 解题思路：升级版leetCode300最长递增子序列
         *  现将信封按照宽度递增排序 宽度相同按照长度递减排序
         *  这样宽度我们就不必再关心了 只要看长度就行了
         *  长度降序保证了不会出现多个相同宽度的信封
         *  然后就是以长度为数组求最长递增子序列问题
         */
        Arrays.sort(envelopes, (c1, c2) -> {
            int first = Integer.compare(c1[0], c2[0]);
            if (first != 0) return first;
            return Integer.compare(c2[1], c1[1]);
        });

        int i = 0;
        int[] nums = new int[envelopes.length];
        for (int[] envelope : envelopes) {
            nums[i++] = envelope[1];
        }

        return lengthOfLISSolveByBinary(nums);

        //因为有变态测例 普通的dp求最长递增子序列会超时 需要采用二分法
        // int result = 1;
        // int[] dp = new int[envelopes.length];

        // dp[0] = 1;
        // for (int i = 1; i < envelopes.length; i++) {
        //     dp[i] = 1;
        //     for (int j = 0; j < i; j++) {
        //         if(envelopes[j][1] < envelopes[i][1]) {
        //             dp[i] = Math.max(dp[i], dp[j] + 1);
        //         }
        //     }
        //     result = Math.max(result, dp[i]);
        // }

        // return result;
    }

    /**
     * 931. 下降路径最小和
     * <p>
     * 给你一个 n x n 的 方形 整数数组 matrix ，请你找出并返回通过 matrix 的下降路径 的 最小和 。
     * <p>
     * 下降路径 可以从第一行中的任何元素开始，并从每一行中选择一个元素。
     * 在下一行选择的元素和当前行所选元素最多相隔一列（即位于正下方或者沿对角线向左或者向右的第一个元素）。
     * 具体来说，位置 (row, col) 的下一个元素应当是 (row + 1, col - 1)、(row + 1, col) 或者 (row + 1, col + 1) 。
     *
     * @param matrix
     * @return
     */
    int[][] memo931;

    public int leetCode931(int[][] matrix) {
        /*
         * 解题思路：
         *  从最底层开始向上累加计算结果 最后遍历最顶层 得到最小结果
         *  base case就是最底层的自身数值
         *  通过memo备忘录 记录已经计算的结果
         */

        //备忘录初始化
        memo931 = new int[matrix.length][matrix.length];
        for (int[] intArr : memo931) {
            Arrays.fill(intArr, Integer.MIN_VALUE);
        }

        //遍历最顶层 寻找最小结果
        int result = Integer.MAX_VALUE;
        //x横坐标 y纵坐标
        for (int x = 0; x < matrix.length; x++) {
            result = Math.min(result, dp931(x, 0, matrix));
        }

        return result;
    }

    private int dp931(int x, int y, int[][] matrix) {
        //base case 最后一层直接返回自身结果
        if (y == matrix.length - 1) {
            return matrix[y][x];
        }

        if (memo931[y][x] == Integer.MIN_VALUE) {
            //最小值结果等于 下一层(y+1)的左中右最小值 + 当前数值
            memo931[y][x] = getMin931
                    (
                            dp931(Math.max(x - 1, 0), y + 1, matrix),
                            dp931(x, y + 1, matrix),
                            dp931(Math.min(x + 1, matrix.length - 1), y + 1, matrix)
                    )
                    + matrix[y][x];
        }

        return memo931[y][x];
    }

    private int getMin931(int left, int mid, int right) {
        return Math.min(Math.min(left, mid), right);
    }

    /**
     * 70. 爬楼梯
     * <p>
     * 假设你正在爬楼梯。需要 n 阶你才能到达楼顶。
     * <p>
     * 每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
     *
     * @param n
     * @return
     */
    public int leetCode70(int n) {
        if (n < 3) return n;

        int lastOne = 1;
        int lastTwo = 2;
        int result = 0;
        for (int i = 3; i <= n; i++) {
            result = lastOne + lastTwo;
            lastOne = lastTwo;
            lastTwo = result;
        }

        return result;
    }

    /**
     * 746. 使用最小花费爬楼梯
     * <p>
     * 给你一个整数数组 cost ，其中 cost[i] 是从楼梯第 i 个台阶向上爬需要支付的费用。
     * 一旦你支付此费用，即可选择向上爬一个或者两个台阶。
     * <p>
     * 你可以选择从下标为 0 或下标为 1 的台阶开始爬楼梯。
     * <p>
     * 请你计算并返回达到楼梯顶部的最低花费。
     *
     * @param cost
     * @return
     */
    public int leetCode746(int[] cost) {
        int n = cost.length;
        int[] dp = new int[n];

        //base case
        dp[0] = cost[0];
        dp[1] = cost[1];

        for (int i = 2; i < n; i++) {
            dp[i] = Math.min(dp[i - 2], dp[i - 1]) + cost[i];
        }

        return Math.min(dp[n - 1], dp[n - 2]);
    }

    /**
     * 62. 不同路径
     * <p>
     * 一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。
     * <p>
     * 机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。
     * <p>
     * 问总共有多少条不同的路径？
     *
     * @param m
     * @param n
     * @return
     */
    public int leetCode62(int m, int n) {
        /*
         * 解题思路：
         *  一个点只可能从它左面过来或者从它上面过来
         *  得出动态转移方程：dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
         *
         *  棋盘最左侧的一列和最上面的一行 只能是竖着一直走或者横着一直走
         *  所以dp[i][0] = 1; dp[0][i] = 1;
         *
         *  base case dp[0][0] = 1; 上面初始值边界时已经包含了
         */
        int[][] dp = new int[m][n];

        for (int i = 0; i < m; i++) {
            dp[i][0] = 1;
        }
        for (int i = 0; i < n; i++) {
            dp[0][i] = 1;
        }

        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }

        return dp[m - 1][n - 1];
    }

    /**
     * 63. 不同路径 II
     * <p>
     * 一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。
     * <p>
     * 机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish”）。
     * <p>
     * 现在考虑网格中有障碍物。那么从左上角到右下角将会有多少条不同的路径？
     * <p>
     * 网格中的障碍物和空位置分别用 1 和 0 来表示。
     *
     * @param obstacleGrid
     * @return
     */
    public int leetCode63(int[][] obstacleGrid) {
        /*
         * 解题思路：leetCode62升级版
         *  注意点
         *  1:初始化的第一行和第一列的时候 出现障碍物后面的全部为0 因为走不到了
         *  2.计算的时候也是 obstacleGrid[i][j]==1时 dp[i][j]=0
         *  因为int的默认值就是0 所以上述两种赋值0的情况 不用写
         */
        int x = obstacleGrid[0].length;
        int y = obstacleGrid.length;
        int[][] dp = new int[y][x];

        for (int i = 0; i < x; i++) {
            if (obstacleGrid[0][i] == 1) break;
            dp[0][i] = 1;
        }

        for (int i = 0; i < y; i++) {
            if (obstacleGrid[i][0] == 1) break;
            dp[i][0] = 1;
        }

        for (int i = 1; i < y; i++) {
            for (int j = 1; j < x; j++) {
                if (obstacleGrid[i][j] == 0) {
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
                }
            }
        }

        return dp[y - 1][x - 1];
    }

    /**
     * 343. 整数拆分
     * <p>
     * 给定一个正整数 n ，将其拆分为 k 个 正整数 的和（ k >= 2 ），并使这些整数的乘积最大化。
     * <p>
     * 返回 你可以获得的最大乘积 。
     *
     * @param n
     * @return
     */
    public int leetCode343(int n) {
        /*
         * 解题思路:
         *  dp[n]代表当前数最大乘积
         *  i可以拆为两个部分:i-j 和 j  (j<i-1)
         *  j * (i - j) 是单纯的把整数拆分为两个数相乘，而j * dp[i - j]是拆分成两个以及两个以上的个数相乘。
         *  dp[i]的最大值是 遍历j并取 两数相乘的结果j*(i-j) 或 j与dp[i-j]的乘积 中最大值的最大值
         *  因为要求每层中最大的结果 所以要遍历j 取最大值 所以是最大值的最大值
         */
        int[] dp = new int[n + 1];

        //base case
        dp[2] = 1;

        for (int i = 3; i <= n; i++) {
            for (int j = 1; j < i - 1; j++) {
                dp[i] = Math.max(dp[i], Math.max((i - j) * j, j * dp[i - j]));
            }
        }

        return dp[n];
    }

    /**
     * 96. 不同的二叉搜索树
     * <p>
     * 给你一个整数 n ，求恰由 n 个节点组成且节点值从 1 到 n 互不相同的 二叉搜索树 有多少种？
     * 返回满足题意的二叉搜索树的种数。
     *
     * @param n
     * @return
     */
    public int leetCode96(int n) {
        /*
         * 解题思路：
         *  dp[i]代表n=i时的结果
         *  由于后续结果肯定要要依赖前面的结果进行计算 所以是自底向上遍历求i
         *  每当i增加时,由于i为最大值 增加方式有两种：
         *      1.在不破坏原有结构的基础上 有两种：
         *          1.1:在i-1的右子树上直接添加
         *          1.2:直接以i为根节点，将原有的树作为左子树添加到i上
         *          综上：情况1的数量为dp[i-1]*2
         *      2.破坏原有的基础 那就说明i必须满足两点
         *          第一点：i的左子树必须要有值（即i的下面）
         *          第二点：i必须为某个值的右子树（即i的上面）
         *          所以：以i为分界点，i的上面有j个元素，下面就有i-1-j个元素（因为i自身占一个元素）
         *               j要小于i-1（因为j=i-1时，i下面就没有元素了，就和1.1的情况一样了）
         *          综上：情况2的数量为遍历j从1到i-2的dp[j] * dp[i - 1 - j]的总和（即上下结果乘积）
         *  最终计算结果就是情况1+情况2的结果
         */
        if (n == 1) return 1;
        int[] dp = new int[n + 1];

        //base case
        dp[1] = 1;
        dp[2] = 2;

        for (int i = 3; i <= n; i++) {
            int sum = 0;
            //计算情况2的总和
            for (int j = 1; j < i - 1; j++) {
                sum += dp[j] * dp[i - 1 - j];
            }
            //最终结果为情况1+情况2的结果
            dp[i] = 2 * dp[i - 1] + sum;
        }

        return dp[n];
    }

    /**
     * 416. 分割等和子集
     * <p>
     * 给你一个 只包含正整数 的 非空 数组 nums 。
     * 请你判断是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。
     *
     * @param nums
     * @return
     */
    public boolean leetCode416(int[] nums) {
        /*
         * 解题思路：
         *  01背包问题 target等于总和除以2
         *  最后结果 就是dp[nums.length - 1][target]是否等于target
         */
        int sum = Arrays.stream(nums).sum();
        //特判 和为奇数 直接返回false
        if (sum % 2 != 0) return false;

        int target = sum / 2;

        int[][] dp = new int[nums.length][target + 1];

        //base case
        for (int j = nums[0]; j <= target; j++) {
            dp[0][j] = nums[0];
        }

        for (int i = 1; i < nums.length; i++) {
            for (int j = 1; j <= target; j++) {
                if (j < nums[i]) {
                    dp[i][j] = dp[i - 1][j];
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i - 1][j - nums[i]] + nums[i]);
                }
            }
        }

        return dp[nums.length - 1][target] == target;
    }

    /**
     * 1049. 最后一块石头的重量 II
     * <p>
     * 有一堆石头，用整数数组 stones 表示。其中 stones[i] 表示第 i 块石头的重量。
     * <p>
     * 每一回合，从中选出任意两块石头，然后将它们一起粉碎。假设石头的重量分别为 x 和 y，且 x <= y。
     * 那么粉碎的可能结果如下：
     * 如果 x == y，那么两块石头都会被完全粉碎；
     * 如果 x != y，那么重量为 x 的石头将会完全粉碎，而重量为 y 的石头新重量为 y-x。
     * 最后，最多只会剩下一块 石头。返回此石头 最小的可能重量 。如果没有石头剩下，就返回 0。
     *
     * @param stones
     * @return
     */
    public int leetCode1049(int[] stones) {
        int sum = Arrays.stream(stones).sum();
        int target = sum / 2;

        int[][] dp = new int[stones.length][target + 1];

        //base case
        for (int j = stones[0]; j <= target; j++) {
            dp[0][j] = stones[0];
        }

        for (int i = 1; i < stones.length; i++) {
            for (int j = 1; j <= target; j++) {
                if (j < stones[i]) {
                    dp[i][j] = dp[i - 1][j];
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i - 1][j - stones[i]] + stones[i]);
                }
            }
        }

        return sum - 2 * dp[stones.length - 1][target];
    }

    /**
     * 494. 目标和
     * <p>
     * 给你一个整数数组 nums 和一个整数 target 。
     * <p>
     * 向数组中的每个整数前添加 '+' 或 '-' ，然后串联起所有整数，可以构造一个 表达式 ：
     * <p>
     * 例如，nums = [2, 1] ，可以在 2 之前添加 '+' ，在 1 之前添加 '-' ，然后串联起来得到表达式 "+2-1" 。
     * 返回可以通过上述方法构造的、运算结果等于 target 的不同 表达式 的数目。
     *
     * @param nums
     * @param target
     * @return
     */
    public int leetCode494(int[] nums, int target) {
        /*
         * 解题思路：
         *  left:正数和;right:负数和
         *      left+right=sum
         *      left-right=target
         *      二元一次求解 left=(sum+target)/2
         *  背包组合问题公式:dp[j]+=dp[j-nums[i]]
         */
        int sum = Math.abs(Arrays.stream(nums).sum());
        //特判
        if ((sum + target) % 2 == 1) return 0;
        if (Math.abs(target) > sum) return 0;

        int size = (sum + target) / 2;
        //dp[j]代表当前填满容量为j的包有多少种方法
        int[] dp = new int[size + 1];

        //base case:填满容量为0的包只有一种方法 就是不填
        dp[0] = 1;

        for (int i = 0; i < nums.length; i++) {
            for (int j = size; j >= nums[i]; j--) {
                dp[j] += dp[j - nums[i]];
            }
        }

        return dp[size];
    }

    /**
     * 474. 一和零
     * <p>
     * 给你一个二进制字符串数组 strs 和两个整数 m 和 n 。
     * <p>
     * 请你找出并返回 strs 的最大子集的长度，该子集中 最多 有 m 个 0 和 n 个 1 。
     * <p>
     * 如果 x 的所有元素也是 y 的元素，集合 x 是集合 y 的 子集 。
     *
     * @param strs
     * @param m
     * @param n
     * @return
     */
    public int leetCode474(String[] strs, int m, int n) {
        int[][][] dp = new int[strs.length][m + 1][n + 1];

        //base case
        int[] baseResult = cacl474(strs[0]);
        for (int i = baseResult[0]; i <= m; i++) {
            for (int j = baseResult[1]; j <= n; j++) {
                dp[0][i][j] = 1;
            }
        }

        for (int i = 1; i < strs.length; i++) {
            int[] resultI = cacl474(strs[i]);
            for (int im = 0; im <= m; im++) {
                for (int in = 0; in <= n; in++) {
                    if (im >= resultI[0] && in >= resultI[1]) {
                        dp[i][im][in] = Math.max(
                                dp[i - 1][im - resultI[0]][in - resultI[1]] + 1,
                                dp[i - 1][im][in]
                        );
                    } else {
                        dp[i][im][in] = dp[i - 1][im][in];
                    }
                }
            }
        }

        return dp[strs.length - 1][m][n];
    }

    private int[] cacl474(String str) {
        int[] result = new int[2];
        for (int i = 0; i < str.length(); i++) {
            if (str.charAt(i) == '0') {
                result[0] += 1;
            } else {
                result[1] += 1;
            }
        }
        return result;
    }

    /**
     * 518. 零钱兑换 II
     * <p>
     * 给你一个整数数组 coins 表示不同面额的硬币，另给一个整数 amount 表示总金额。
     * <p>
     * 请你计算并返回可以凑成总金额的硬币组合数。如果任何硬币组合都无法凑出总金额，返回 0 。
     * <p>
     * 假设每一种面额的硬币有无限个。 
     * <p>
     * 题目数据保证结果符合 32 位带符号整数。
     *
     * @param amount
     * @param coins
     * @return
     */
    public int leetCode518(int amount, int[] coins) {
        /*
         * 解题思路：完全背包问题之组合问题
         *  但是先遍历金额还是先遍历金币要想清楚
         *  先遍历金币是组合问题（即不区分顺序1,2和2,1是一种答案） 先遍历金额是排序问题(区分顺序)
         *  本题要先遍历金币 因为1,2和2,1对于本题来说是一种情况
         *  leetCode377则是先遍历先遍历金额
         */
        //递推表达式
        int[] dp = new int[amount + 1];
        //初始化dp数组，表示金额为0时只有一种情况，也就是什么都不装
        dp[0] = 1;
        for (int i = 0; i < coins.length; i++) {
            for (int j = coins[i]; j <= amount; j++) {
                dp[j] += dp[j - coins[i]];
                System.out.printf("i=%s,j=%s,dp[%s]=dp[%s]+dp[%s],result=%s%n",
                        i, j, j, j, j - coins[i], dp[j]);
            }
        }
        return dp[amount];
    }

    /**
     * 377. 组合总和 Ⅳ
     * <p>
     * 给你一个由 不同 整数组成的数组 nums ，和一个目标整数 target 。
     * 请你从 nums 中找出并返回总和为 target 的元素组合的个数。
     * <p>
     * 题目数据保证答案符合 32 位整数范围。
     *
     * @param nums
     * @param target
     * @return
     */
    public int leetCode377(int[] nums, int target) {
        /*
         * 解题思路：完全背包问题之排列问题
         * 具体分析见leetCode518
         * 大神分析：爬楼梯问题 楼梯的阶数一共为target，一次可以走的步数为nums[i]。 一共有多少种走法？
         */
        int[] dp = new int[target + 1];

        dp[0] = 1;

        for (int j = 1; j <= target; j++) {
            for (int num : nums) {
                if (j >= num) {
                    dp[j] += dp[j - num];
                }
            }
        }

        return dp[target];
    }

    /**
     * 279. 完全平方数
     * <p>
     * 给你一个整数 n ，返回 和为 n 的完全平方数的最少数量 。
     * <p>
     * 完全平方数 是一个整数，其值等于另一个整数的平方；换句话说，其值等于一个整数自乘的积。
     * 例如，1、4、9 和 16 都是完全平方数，而 3 和 11 不是。
     *
     * @param n
     * @return
     */
    public int leetCode279(int n) {
        if (n == 1 || n == 10000) return 1;

        int[] dp = new int[n + 1];
        Arrays.fill(dp, Integer.MAX_VALUE);

        //base case
        dp[0] = 0;

        for (int i = 1; i * i <= n; i++) {
            int square = i * i;
            for (int j = square; j <= n; j++) {
                //剪枝判断 可以没有
                if (dp[j - square] != Integer.MAX_VALUE) {
                    dp[j] = Math.min(dp[j], dp[j - square] + 1);
                }
            }
        }

        return dp[n];
    }

    /**
     * 139. 单词拆分
     * <p>
     * 给你一个字符串 s 和一个字符串列表 wordDict 作为字典。请你判断是否可以利用字典中出现的单词拼接出 s 。
     * <p>
     * 注意：不要求字典中出现的单词全部都使用，并且字典中的单词可以重复使用。
     *
     * @param s
     * @param wordDict
     * @return
     */
    public boolean leetCode139(String s, List<String> wordDict) {
        int len = s.length();
        boolean[] dp = new boolean[len + 1];
        dp[0] = true;

        for (int j = 1; j <= len; j++) {
            for (String word : wordDict) {
                int wordLen = word.length();
                if (j >= wordLen
                        && dp[j - wordLen]
                        && s.substring(j - wordLen, j).equals(word)) {
                    dp[j] = true;
                }
            }
        }

        return dp[len];
    }

    /**
     * 198. 打家劫舍
     * <p>
     * 你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，
     * 影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，
     * 如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。
     * <p>
     * 给定一个代表每个房屋存放金额的非负整数数组，计算你 不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额。
     *
     * @param nums
     * @return
     */
    public int leetCode198(int[] nums) {
        //常规解题方案
        int len = nums.length;
        if (len == 0) return 0;
        if (len == 1) return nums[0];

        int[] dp = new int[len];
        //base case
        dp[0] = nums[0];
        dp[1] = Math.max(nums[0], nums[1]);

        for (int i = 2; i < len; i++) {
            dp[i] = Math.max(dp[i - 1], dp[i - 2] + nums[i]);
        }

        return dp[len - 1];

        //滚动数组 因为当前结果一直依赖前两个结果 所以只需要记住前两个结果的值就好了 然后替换前两个值
        //return rob213(nums, 0, len - 1);
    }

    /**
     * 213. 打家劫舍 II
     * <p>
     * 你是一个专业的小偷，计划偷窃沿街的房屋，每间房内都藏有一定的现金。
     * 这个地方所有的房屋都 围成一圈 ，这意味着第一个房屋和最后一个房屋是紧挨着的。
     * 同时，相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警 。
     * <p>
     * 给定一个代表每个房屋存放金额的非负整数数组，计算你 在不触动警报装置的情况下 ，今晚能够偷窃到的最高金额。
     *
     * @param nums
     * @return
     */
    public int leetCode213(int[] nums) {
        /*
         * 解题思路：leetCode198升级版
         *  最终结果为 去掉第一个元素的leetCode198和去掉最后一个元素的leetCode198的最大值
         */
        int len = nums.length;
        if (len == 0) return 0;
        if (len == 1) return nums[0];
        if (len == 2) return Math.max(nums[0], nums[1]);

        return Math.max(
                rob213(nums, 0, len - 2),
                rob213(nums, 1, len - 1)
        );
    }

    private int rob213(int[] nums, int start, int end) {
        int first = nums[start];
        int second = Math.max(nums[start], nums[start + 1]);

        for (int i = start + 2; i <= end; i++) {
            int temp = Math.max(second, first + nums[i]);
            first = second;
            second = temp;
        }

        return second;
    }

    /**
     * 337. 打家劫舍 III
     * <p>
     * 小偷又发现了一个新的可行窃的地区。这个地区只有一个入口，我们称之为 root 。
     * <p>
     * 除了 root 之外，每栋房子有且只有一个“父“房子与之相连。一番侦察之后，
     * 聪明的小偷意识到“这个地方的所有房屋的排列类似于一棵二叉树”。
     * 如果 两个直接相连的房子在同一天晚上被打劫 ，房屋将自动报警。
     * <p>
     * 给定二叉树的 root 。返回 在不触动警报的情况下 ，小偷能够盗取的最高金额 。
     *
     * @param root
     * @return
     */
    public int leetCode337(TreeNode root) {
        /*
         * 解题思路：二叉树+动态规划
         *  dp定义：选择当前节点的最大值 和 不选择当前节点的最大值 所以需要一个二维数组去存储
         *      选择当前节点的最大值=当前节点的值+左子树不选择当前节点的值+右子树不选择当期节点的值
         *      不选择当前节点的最大值=左子树两种情况的最大值+右子树两种情况的最大值
         *          因为不选择当前节点，左右子树可以选但不是必须选 所以选择左右子树的两种情况的最大值来计算
         *  因为需要二叉树递归产生的结果 所以选择后续遍历
         */
        int[] result = rob337(root);
        return Math.max(result[0], result[1]);
    }

    private int[] rob337(TreeNode root) {
        int[] result = new int[2];
        if (root == null) return result;

        int[] leftResult = rob337(root.left);
        int[] rightResult = rob337(root.right);

        //选择当前结点 = 不选左子树 + 不选右子树 + 当前结点值
        result[1] = leftResult[0] + rightResult[0] + root.val;
        //不选择当前结点 = max(左子树两种情况) + max(右子树两种情况)
        result[0] = Math.max(leftResult[0], leftResult[1]) + Math.max(rightResult[0], rightResult[1]);

        return result;
    }

    /**
     * 121. 买卖股票的最佳时机
     * 给定一个数组 prices ，它的第 i 个元素 prices[i] 表示一支给定股票第 i 天的价格。
     * <p>
     * 你只能选择 某一天 买入这只股票，并选择在 未来的某一个不同的日子 卖出该股票。
     * 设计一个算法来计算你所能获取的最大利润。
     * <p>
     * 返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 0 。
     *
     * @param prices
     * @return
     */
    public int leetCode121(int[] prices) {
        /*
         * 解题思路：动态规划dp[i][0]、dp[i][1]
         *  dp[i][0]=第i天没持有股票的最大利润
         *      如果昨天也没持有 就等于昨天dp[i-1][0]
         *      如果昨天持有了 今天每持有说明今天卖了 所以等于dp[i-1][1]+prices[i]
         *      最后取结果大的作为dp[i][0]的结果
         *  dp[i][1]=第i天持有股票的最大利润
         *      如果昨天没持有 说明今天刚买 但只能买一次 所以等于0-prices[i]
         *      如果昨天也持有了 就等于昨天dp[i - 1][1]
         *  最后结果就是最后一天没持有的结果 因为持有不卖肯定是负数
         */
        /*int[][] dp = new int[prices.length][2];

        //base case
        dp[0][0] = 0;
        dp[0][1] = -prices[0];

        for (int i = 1; i < prices.length; i++) {
            dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1] + prices[i]);
            dp[i][1] = Math.max(dp[i - 1][1], -prices[i]);
        }

        return dp[prices.length - 1][0];*/

        //因为上述过程只和前一天结果有关 可以用滚动数组
        int[] dp = new int[2];

        dp[1] = -prices[0];

        for (int i = 1; i < prices.length; i++) {
            dp[0] = Math.max(dp[0], dp[1] + prices[i]);
            dp[1] = Math.max(dp[1], -prices[i]);
        }

        return dp[0];
    }

    /**
     * 122. 买卖股票的最佳时机 II
     * <p>
     * 给你一个整数数组 prices ，其中 prices[i] 表示某支股票第 i 天的价格。
     * <p>
     * 在每一天，你可以决定是否购买和/或出售股票。你在任何时候 最多 只能持有 一股 股票。
     * 你也可以先购买，然后在 同一天 出售。
     * <p>
     * 返回 你能获得的 最大 利润 。
     *
     * @param prices
     * @return
     */
    public int leetCode122(int[] prices) {
        /*
         * 解题思路：leetCode121升级版
         *  与121不同的是 可以多次购买
         *  所以dp[i][1]就是在原有已经盈利的基础上再去减去今天的股票价钱即dp[i][0] - prices[i];
         *  最后因为也是依赖前一天的数据 所以可以转换成滚动数组
         */
        /*int[][] dp = new int[prices.length][2];

        dp[0][0] = 0;
        dp[0][1] = -prices[0];

        for (int i = 1; i < prices.length; i++) {
            dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1] + prices[i]);
            dp[i][1] = Math.max(dp[i - 1][1], dp[i - 1][0] - prices[i]);
        }

        return dp[prices.length - 1][0]*/
        ;

        int[] dp = new int[2];

        //base case
        dp[1] = -prices[0];

        for (int i = 1; i < prices.length; i++) {
            dp[0] = Math.max(dp[0], dp[1] + prices[i]);
            dp[1] = Math.max(dp[1], dp[0] - prices[i]);
        }

        return dp[0];
    }

    /**
     * 123. 买卖股票的最佳时机 III
     * <p>
     * 给定一个数组，它的第 i 个元素是一支给定的股票在第 i 天的价格。
     * <p>
     * 设计一个算法来计算你所能获取的最大利润。你最多可以完成 两笔 交易。
     * <p>
     * 注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
     *
     * @param prices
     * @return
     */
    public int leetCode123(int[] prices) {
        /*
         * 解题思路：
         *  dp数据第二位表示4种状态
         *      状态1:第一次买入
         *      状态2:第一次卖出
         *      状态3:第二次买入
         *      状态4:第二次卖出
         *  初始化：
         *      第一次买入就是-prices[0]
         *      第二次买入也是-prices[0] 很重要
         *          在第一次交易取到收益之前 dp[状态2]和dp[状态4]是一样的
         *          如果未初始化 则dp[状态4]处于只卖出的钱 没计算本金
         */
        /*int len = prices.length;
        int[][] dp = new int[len][5];

        //base case
        dp[0][1] = -prices[0];
        dp[0][3] = -prices[0];

        for (int i = 1; i < len; i++) {
            dp[i][1] = Math.max(dp[i - 1][1], -prices[i]);
            dp[i][2] = Math.max(dp[i - 1][2], dp[i - 1][1] + prices[i]);
            dp[i][3] = Math.max(dp[i - 1][3], dp[i - 1][2] - prices[i]);
            dp[i][4] = Math.max(dp[i - 1][4], dp[i - 1][3] + prices[i]);
        }

        return dp[len - 1][4];*/

        //滚动数组实现
        int[] dp = new int[4];

        //base case
        dp[0] = -prices[0];
        dp[2] = -prices[0];

        for (int i = 1; i < prices.length; i++) {
            dp[0] = Math.max(dp[0], -prices[i]);
            dp[1] = Math.max(dp[1], dp[0] + prices[i]);
            dp[2] = Math.max(dp[2], dp[1] - prices[i]);
            dp[3] = Math.max(dp[3], dp[2] + prices[i]);
        }

        return dp[3];
    }

    /**
     * 188. 买卖股票的最佳时机 IV
     * <p>
     * 给定一个整数数组 prices ，它的第 i 个元素 prices[i] 是一支给定的股票在第 i 天的价格。
     * <p>
     * 设计一个算法来计算你所能获取的最大利润。你最多可以完成 k 笔交易。
     * <p>
     * 注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
     *
     * @param k
     * @param prices
     * @return
     */
    public int leetCode188(int k, int[] prices) {
        /*
         * 解题思路：leetCode123升级版
         *  dp数组表示状态,有k比交易就有2k个状态（买入和卖出）双数买入、单数卖出
         *  第一天买入是没有前置盈利的 所以dp[0]=Math.max(dp[0], -prices[i])
         *  其余每次买入都要计算之前的盈利 所以dp[j] = Math.max(dp[j], dp[j - 1] - prices[i]);
         *  每次卖出dp[j] = Math.max(dp[j], dp[j - 1] + prices[i])
         */
        int kLen = 2 * k;
        int[] dp = new int[kLen];

        int ori = -prices[0];
        for (int i = 0; i < kLen; i++) {
            if (i % 2 == 0) dp[i] = ori;
        }

        for (int i = 1; i < prices.length; i++) {
            dp[0] = Math.max(dp[0], -prices[i]);
            for (int j = 1; j < kLen; j++) {
                if (j % 2 == 0) {
                    dp[j] = Math.max(dp[j], dp[j - 1] - prices[i]);
                } else {
                    dp[j] = Math.max(dp[j], dp[j - 1] + prices[i]);
                }
            }
        }

        return dp[kLen - 1];
    }

    /**
     * 309. 最佳买卖股票时机含冷冻期
     * <p>
     * 给定一个整数数组prices，其中第  prices[i] 表示第 i 天的股票价格 。​
     * <p>
     * 设计一个算法计算出最大利润。在满足以下约束条件下，你可以尽可能地完成更多的交易（多次买卖一支股票）:
     * <p>
     * 卖出股票后，你无法在第二天买入股票 (即冷冻期为 1 天)。
     * 注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
     *
     * @param prices
     * @return
     */
    public int leetCode309(int[] prices) {
        int[] dp = new int[4];

        dp[0] = -prices[0];
        dp[1] = 0;
        for (int i = 1; i <= prices.length; i++) {
            // 使用临时变量来保存dp[0], dp[2]
            // 因为马上dp[0]和dp[2]的数据都会变
            int temp = dp[0];
            int temp1 = dp[2];
            dp[0] = Math.max(dp[0], Math.max(dp[3], dp[1]) - prices[i - 1]);
            dp[1] = Math.max(dp[1], dp[3]);
            dp[2] = temp + prices[i - 1];
            dp[3] = temp1;
        }
        return Math.max(dp[3], Math.max(dp[1], dp[2]));
    }

    /**
     * 714. 买卖股票的最佳时机含手续费
     * <p>
     * 给定一个整数数组 prices，其中 prices[i]表示第 i 天的股票价格 ；整数 fee 代表了交易股票的手续费用。
     * <p>
     * 你可以无限次地完成交易，但是你每笔交易都需要付手续费。如果你已经购买了一个股票，
     * 在卖出它之前你就不能再继续购买股票了。
     * <p>
     * 返回获得利润的最大值。
     * <p>
     * 注意：这里的一笔交易指买入持有并卖出股票的整个过程，每笔交易你只需要为支付一次手续费。
     *
     * @param prices
     * @param fee
     * @return
     */
    public int leetCode714(int[] prices, int fee) {
        /*
         * 解题思路：leetCode122升级版
         *  在leetCode122的基础上 增加了卖出时候扣除手续费
         */
        int[] dp = new int[2];

        dp[1] = -prices[0];

        for (int i = 1; i < prices.length; i++) {
            dp[0] = Math.max(dp[0], dp[1] + prices[i] - fee);
            dp[1] = Math.max(dp[1], dp[0] - prices[i]);
        }

        return dp[0];
    }

    /**
     * 674. 最长连续递增序列
     * <p>
     * 给定一个未经排序的整数数组，找到最长且 连续递增的子序列，并返回该序列的长度。
     * <p>
     * 连续递增的子序列 可以由两个下标 l 和 r（l < r）确定，如果对于每个 l <= i < r，都有 nums[i] < nums[i + 1] ，
     * 那么子序列 [nums[l], nums[l + 1], ..., nums[r - 1], nums[r]] 就是连续递增子序列。
     *
     * @param nums
     * @return
     */
    public int leetCode674(int[] nums) {
        /*
         * 解题思路：
         *  当前比前一个值大就计数加一，比前一个值小则重新计数
         *  pre记录上一个值，temp代表计数
         */
        int result = 1;
        int temp = 0;
        int pre = Integer.MIN_VALUE;

        for (int i = 0; i < nums.length; i++) {
            if (nums[i] > pre) {
                result = Math.max(result, ++temp);
            } else {
                temp = 1;
            }
            pre = nums[i];
        }

        return result;
    }

    /**
     * 718. 最长重复子数组
     * <p>
     * 给两个整数数组 nums1 和 nums2 ，返回 两个数组中 公共的 、长度最长的子数组的长度 。
     *
     * @param nums1
     * @param nums2
     * @return
     */
    public int leetCode718(int[] nums1, int[] nums2) {
        /*
         * 解题思路：参考算法笔记 动态规划.子序列问题
         */
        int result = 0;
        int[][] dp = new int[nums1.length + 1][nums2.length + 1];

        for (int i = 0; i < nums1.length; i++) {
            for (int j = 0; j < nums2.length; j++) {
                if (nums1[i] == nums2[j]) {
                    dp[i + 1][j + 1] = dp[i][j] + 1;
                    result = Math.max(result, dp[i + 1][j + 1]);
                }
            }
        }

        return result;
    }

    /**
     * 1143. 最长公共子序列
     * <p>
     * 给定两个字符串 text1 和 text2，返回这两个字符串的最长 公共子序列 的长度。如果不存在 公共子序列 ，返回 0 。
     * <p>
     * 一个字符串的 子序列 是指这样一个新的字符串：
     * 它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。
     * <p>
     * 例如，"ace" 是 "abcde" 的子序列，但 "aec" 不是 "abcde" 的子序列。
     * 两个字符串的 公共子序列 是这两个字符串所共同拥有的子序列。
     *
     * @param text1
     * @param text2
     * @return
     */
    public int leetCode1143(String text1, String text2) {
        /*
         * 解题思路：参考算法笔记 动态规划.子序列问题
         */
        int len1 = text1.length();
        int len2 = text2.length();
        int[][] dp = new int[len1 + 1][len2 + 1];

        for (int i = 1; i <= len1; i++) {
            for (int j = 1; j <= len2; j++) {
                if (text1.charAt(i - 1) == text2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }

        return dp[len1][len2];
    }

    /**
     * 1035. 不相交的线
     * <p>
     * 在两条独立的水平线上按给定的顺序写下 nums1 和 nums2 中的整数。
     * <p>
     * 现在，可以绘制一些连接两个数字 nums1[i] 和 nums2[j] 的直线，这些直线需要同时满足满足：
     * <p>
     *  nums1[i] == nums2[j]
     * 且绘制的直线不与任何其他连线（非水平线）相交。
     * 请注意，连线即使在端点也不能相交：每个数字只能属于一条连线。
     * <p>
     * 以这种方法绘制线条，并返回可以绘制的最大连线数。
     *
     * @param nums1
     * @param nums2
     * @return
     */
    public int leetCode1035(int[] nums1, int[] nums2) {
        /*
         * 解题思路:
         *  这道题和leetCode1143(最长公共子序列)解法一模一样
         *  不相交其实就是公共子串 因为子串是按顺序的 所以不会相交
         */
        int len1 = nums1.length;
        int len2 = nums2.length;
        int[][] dp = new int[len1 + 1][len2 + 1];

        for (int i = 1; i <= len1; i++) {
            for (int j = 1; j <= len2; j++) {
                if (nums1[i - 1] == nums2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i][j - 1], dp[i - 1][j]);
                }
            }
        }

        return dp[len1][len2];
    }

    /**
     * 53. 最大子数组和
     * <p>
     * 给你一个整数数组 nums ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
     * <p>
     * 子数组 是数组中的一个连续部分。
     *
     * @param nums
     * @return
     */
    public int leetCode53(int[] nums) {
        /*
         * 解题思路：
         *  依次叠加每个数并记录和sum
         *  当遍历到i时，如果sum+i还不如i自身大
         *  那就说明前面的数都是累赘 直接从当前i作为第一个叠加元素继续向后遍历
         */
        int result = nums[0];
        int sum = nums[0];

        for (int i = 1; i < nums.length; i++) {
            sum = Math.max(sum + nums[i], nums[i]);
            result = Math.max(sum, result);
        }

        return result;
    }

    /**
     * 392. 判断子序列
     * <p>
     * 给定字符串 s 和 t ，判断 s 是否为 t 的子序列。
     * <p>
     * 字符串的一个子序列是原始字符串删除一些（也可以不删除）字符而不改变剩余字符相对位置形成的新字符串。
     * （例如，"ace"是"abcde"的一个子序列，而"aec"不是）。
     * <p>
     * 进阶：
     * <p>
     * 如果有大量输入的 S，称作 S1, S2, ... , Sk 其中 k >= 10亿，你需要依次检查它们是否为 T 的子序列。
     * 在这种情况下，你会怎样改变代码？
     *
     * @param s
     * @param t
     * @return
     */
    public boolean leetCode392(String s, String t) {
        /*
         * 解题思路:
         *  不考虑进阶问题 直接双指针就可以
         *  考虑进阶问题 其实就是最长重复子数组问题(leetCode1143)
         */
        int len1 = s.length();
        int len2 = t.length();
        int[][] dp = new int[len1 + 1][len2 + 1];

        for (int i = 1; i <= len1; i++) {
            for (int j = 1; j <= len2; j++) {
                if (s.charAt(i - 1) == t.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    //leetCode1143这里是dp[i][j] = Math.max(dp[i][j - 1], dp[i - 1][j]);
                    //不同点在于1143不知道谁是谁的子序列 此题知道s是t的子序列
                    dp[i][j] = dp[i][j - 1];
                }
            }
        }

        return dp[len1][len2] == len1;

        //双指针
        /*if (s.length() == 0) return true;
        if (t.length() == 0) return false;

        int j = 0;
        for (int i = 0; i < t.length(); i++) {
            if (s.charAt(j) == t.charAt(i)) {
                if (j == s.length() - 1) return true;
                j++;
            }
        }

        return false;*/
    }

    /**
     * 115. 不同的子序列
     * <p>
     * 给定一个字符串 s 和一个字符串 t ，计算在 s 的子序列中 t 出现的个数。
     * <p>
     * 字符串的一个 子序列 是指，通过删除一些（也可以不删除）字符且不干扰剩余字符相对位置所组成的新字符串。
     * （例如，"ACE" 是 "ABCDE" 的一个子序列，而 "AEC" 不是）
     * <p>
     * 题目数据保证答案符合 32 位带符号整数范围。
     *
     * @param s
     * @param t
     * @return
     */
    public int leetCode115(String s, String t) {
        int len1 = s.length();
        int len2 = t.length();
        int[][] dp = new int[len1 + 1][len2 + 1];

        //base case 子串为空时 结果都是1
        for (int i = 0; i <= len1; i++) {
            dp[i][0] = 1;
        }

        for (int i = 1; i <= len1; i++) {
            for (int j = 1; j <= len2; j++) {
                if (s.charAt(i - 1) == t.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j];
                } else {
                    dp[i][j] = dp[i - 1][j];
                }
            }
        }

        return dp[len1][len2];
    }

    /**
     * 583. 两个字符串的删除操作
     * <p>
     * 给定两个单词 word1 和 word2 ，返回使得 word1 和  word2 相同所需的最小步数。
     * <p>
     * 每步 可以删除任意一个字符串中的一个字符。
     *
     * @param word1
     * @param word2
     * @return
     */
    public int leetCode583(String word1, String word2) {
        /*
         * 解题思路：就是求最长公共子序列 leetCode1143
         */
        int len1 = word1.length();
        int len2 = word2.length();
        int[][] dp = new int[len1 + 1][len2 + 1];

        for (int i = 1; i <= len1; i++) {
            for (int j = 1; j <= len2; j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }

        return len1 + len2 - 2 * dp[len1][len2];
    }

    /**
     * 72. 编辑距离
     * <p>
     * 给你两个单词 word1 和 word2， 请返回将 word1 转换成 word2 所使用的最少操作数  。
     * <p>
     * 你可以对一个单词进行如下三种操作：
     * <p>
     * 插入一个字符
     * 删除一个字符
     * 替换一个字符
     *
     * @param word1
     * @param word2
     * @return
     */
    public int leetCode72(String word1, String word2) {
        /*
         * 解题思路：
         *  可以套用子序问题模板
         *  字符相等时，由于不用操作 所以dp[i][j] = dp[i - 1][j - 1]
         *  字符不相等时，分为增删改三种情况
         *      删除：在删除word1或者word2一个字符基础上加一（即本次操作）则为当前结果
         *           dp[i][j] = dp[i][j - 1] + 1 或 dp[i - 1][j] + 1
         *      增加：增加和删除是一样的，给word1增加字符和给word2删除字符是一个效果
         *      修改：修改则是将当前两个字符修改为一样的
         *           dp[i][j] = dp[i - 1][j - 1] + 1
         *      综上所述：取三种情况最小的
         *  初始化：当一个字符串为空时，最小操作就是逐一删除另一个字符串的所有字符
         */
        int len1 = word1.length();
        int len2 = word2.length();
        int[][] dp = new int[len1 + 1][len2 + 1];

        //base case
        for (int i = 0; i <= len1; i++) {
            dp[i][0] = i;
        }
        for (int j = 0; j <= len2; j++) {
            dp[0][j] = j;
        }

        for (int i = 1; i <= len1; i++) {
            for (int j = 1; j <= len2; j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = Math.min(
                            Math.min(dp[i][j - 1] + 1, dp[i - 1][j] + 1),
                            dp[i - 1][j - 1] + 1
                    );
                }
            }
        }

        return dp[len1][len2];
    }

    /**
     * 647. 回文子串
     * <p>
     * 给你一个字符串 s ，请你统计并返回这个字符串中 回文子串 的数目。
     * <p>
     * 回文字符串 是正着读和倒过来读一样的字符串。
     * <p>
     * 子字符串 是字符串中的由连续字符组成的一个序列。
     * <p>
     * 具有不同开始位置或结束位置的子串，即使是由相同的字符组成，也会被视作不同的子串。
     *
     * @param s
     * @return
     */
    public int leetCode647(String s) {
        /*
         * 解题思路：中心扩展（下文详解） 或动态规划（参考代码随想录）
         *  依次将每个字符作为回文字符串的中心 向两侧扩张
         *  每扩张一次 如果相等则结果加一 不等则退出 计算下一个字符
         *  回文字符串的中心有两种情况：
         *      长度为奇数：那么当前字符作为中心向两侧扩展
         *      长度为偶数：当前字符加上下一个字符作为中心向两次扩展
         *  可以把两种情况写成一种写法：左指针不变，右指针一次等于左指针一次等于左指针加一
         *      左指针left = i / 2
         *      右指针right = left + i % 2
         */
        int result = 0;
        int len = s.length();

        //因为存在两种情况 所以i要遍历到2 * len
        for (int i = 0; i <= 2 * len; i++) {
            //每两步前进一次 可以保证两种情况
            int left = i / 2;
            //区分两种情况 等于左指针和左指针加一
            int right = left + i % 2;

            while (left >= 0 && right < len && s.charAt(left) == s.charAt(right)) {
                result++;
                left--;
                right++;
            }
        }

        return result;
    }

    /**
     * 516. 最长回文子序列
     * <p>
     * 给你一个字符串 s ，找出其中最长的回文子序列，并返回该序列的长度。
     * <p>
     * 子序列定义为：不改变剩余字符顺序的情况下，删除某些字符或者不删除任何字符形成的一个序列。
     *
     * @param s
     * @return
     */
    public int leetCode516(String s) {
        /*
         * 解题思路：
         *  dp含义：字符串i到j的最长回文子序列
         *  dp方程：
         *      当i和j的字符相等时 结果等于不包含i和j的字符串的结果加2（加的是i和j）
         *      不相等时，结果向左进一位或者向右退一位的最大值
         *  dp初始化：dp[i][i]都为1 因为自身就是一个回文串
         *  遍历顺序：因为dp[i][j]依赖于i+1和j-1 所以i到倒序遍历 j正序遍历
         *  最后的结果就是整个字符串的结果 也就是从0到len-1
         */
        int len = s.length();
        int[][] dp = new int[len + 1][len + 1];

        for (int i = len - 1; i >= 0; i--) {
            //base case
            dp[i][i] = 1;

            for (int j = i + 1; j < len; j++) {
                if (s.charAt(i) == s.charAt(j)) {
                    dp[i][j] = dp[i + 1][j - 1] + 2;
                } else {
                    dp[i][j] = Math.max(dp[i][j - 1], dp[i + 1][j]);
                }
            }
        }

        return dp[0][len - 1];
    }
}
