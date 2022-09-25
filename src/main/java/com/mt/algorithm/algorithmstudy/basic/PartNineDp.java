package com.mt.algorithm.algorithmstudy.basic;

import org.springframework.stereotype.Service;

import java.util.Arrays;

/**
 * @Description
 * @Author T
 * @Date 2022/9/1
 */
@Service
public class PartNineDp {

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
         * 解题思路：完全背包问题
         *  但是先遍历金额还是先遍历金币要想清楚
         *  本题要先遍历金币 因为1,2和2,1对于本题来说是一种情况
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

}
