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
}
