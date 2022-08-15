package com.mt.algorithm.algorithmstudy.basic;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.stream.Collectors;

/**
 * @Description
 * @Author T
 * @Date 2022/8/1
 */
@Service
public class PartSixBinaryTree {

    /**
     * Definition for a TreeNode.
     */
    @AllArgsConstructor
    @NoArgsConstructor
    @Data
    static class TreeNode {
        private int val;
        private TreeNode left;
        private TreeNode right;

        public TreeNode(int val) {
            this.val = val;
        }
    }

    /**
     * Definition for a Node.
     */
    @AllArgsConstructor
    @NoArgsConstructor
    @Data
    class Node {
        public int val;
        public List<Node> children;

        public Node(int _val) {
            val = _val;
        }
    }

    /**
     * 144. 二叉树的前序遍历
     *
     * @param root
     * @return
     */
    public List<Integer> leetCode144(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        preorderTraversal(root, result);
        return result;
    }

    private void preorderTraversal(TreeNode root, List<Integer> result) {
        if (root == null) {
            return;
        }
        result.add(root.val);
        preorderTraversal(root.left, result);
        preorderTraversal(root.right, result);
    }

    /**
     * 102. 二叉树的层序遍历
     *
     * @param root
     * @return
     */
    public List<List<Integer>> leetCode102(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();

        if (root == null) {
            return result;
        }

        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);

        while (!queue.isEmpty()) {
            List<Integer> curList = new ArrayList<>();
            int len = queue.size();

            while (len > 0) {
                TreeNode curNode = queue.poll();
                curList.add(curNode.val);

                if (curNode.left != null) queue.add(curNode.left);
                if (curNode.right != null) queue.add(curNode.right);

                len--;
            }

            result.add(curList);
        }

        return result;
    }

    /**
     * 107. 二叉树的层序遍历 II
     * <p>
     * 给你二叉树的根节点 root ，返回其节点值 自底向上的层序遍历 。 （即按从叶子节点所在层到根节点所在的层，逐层从左向右遍历）
     *
     * @param root
     * @return
     */
    public List<List<Integer>> leetCode107(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();

        if (root == null) return result;

        Queue<TreeNode> queue = new LinkedList<>();
        Stack<List<Integer>> stack = new Stack<>();
        queue.offer(root);

        while (!queue.isEmpty()) {
            List<Integer> curList = new ArrayList<>();
            int len = queue.size();

            for (int i = 0; i < len; i++) {
                TreeNode curNode = queue.poll();
                curList.add(curNode.val);

                if (curNode.left != null) queue.offer(curNode.left);
                if (curNode.right != null) queue.offer(curNode.right);
            }

            stack.push(curList);
        }

        while (stack.size() > 0) {
            result.add(stack.pop());
        }

        return result;
    }

    /**
     * 199. 二叉树的右视图
     * <p>
     * 给定一个二叉树的 根节点 root，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。
     *
     * @param root
     * @return
     */
    public List<Integer> leetCode199(TreeNode root) {
        List<Integer> result = new ArrayList<>();

        if (root == null) {
            return result;
        }

        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);

        while (!queue.isEmpty()) {
            List<Integer> curList = new ArrayList<>();
            int len = queue.size();

            for (int i = 0; i < len; i++) {
                TreeNode curNode = queue.poll();
                curList.add(curNode.val);

                if (curNode.right != null) queue.offer(curNode.right);
                if (curNode.left != null) queue.offer(curNode.left);
            }

            result.add(curList.get(0));
        }

        return result;
    }

    /**
     * 637. 二叉树的层平均值
     * <p>
     * 给定一个非空二叉树的根节点 root , 以数组的形式返回每一层节点的平均值。与实际答案相差 10-5 以内的答案可以被接受。
     *
     * @param root
     * @return
     */
    public List<Double> leetCode637(TreeNode root) {
        List<Double> result = new ArrayList<>();

        if (root == null) {
            return result;
        }

        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);

        while (!queue.isEmpty()) {
            int len = queue.size();
            double sum = 0;
            for (int i = 0; i < len; i++) {
                TreeNode curNode = queue.poll();
                sum += curNode.val;
                if (curNode.left != null) queue.offer(curNode.left);
                if (curNode.right != null) queue.offer(curNode.right);
            }

            result.add(sum / len);
        }

        return result;
    }

    /**
     * 429. N 叉树的层序遍历
     *
     * @param root
     * @return
     */
    public List<List<Integer>> leetCode429(Node root) {
        List<List<Integer>> result = new ArrayList<>();

        if (root == null) {
            return result;
        }

        Queue<Node> queue = new LinkedList<>();
        queue.offer(root);

        while (!queue.isEmpty()) {
            List<Integer> curList = new ArrayList<>();
            int len = queue.size();

            for (int i = 0; i < len; i++) {
                Node curNode = queue.poll();
                curList.add(curNode.val);
                List<Node> children = curNode.children;
                if (children != null && children.size() > 0) {
                    for (Node node : children) {
                        queue.offer(node);
                    }
                }
            }

            result.add(curList);
        }

        return result;
    }

    /**
     * 515. 在每个树行中找最大值
     * <p>
     * 给定一棵二叉树的根节点 root ，请找出该二叉树中每一层的最大值。
     *
     * @param root
     * @return
     */
    public List<Integer> leetCode515(TreeNode root) {
        if (root == null) {
            return new ArrayList<>();
        }
        List<Integer> res = new ArrayList<>();
        dfs(res, root, 0);
        return res;
    }

    private void dfs(List<Integer> res, TreeNode root, int curHeight) {
        if (curHeight == res.size()) {
            res.add(root.val);
        } else {
            res.set(curHeight, Math.max(res.get(curHeight), root.val));
        }
        if (root.left != null) {
            dfs(res, root.left, curHeight + 1);
        }
        if (root.right != null) {
            dfs(res, root.right, curHeight + 1);
        }
    }

    @Data
    @AllArgsConstructor
    @NoArgsConstructor
    class Node116 {
        public int val;
        public Node116 left;
        public Node116 right;
        public Node116 next;

        public Node116(int _val) {
            val = _val;
        }
    }

    ;

    /**
     * 116. 填充每个节点的下一个右侧节点指针
     * 与 leetCode117.填充每个节点的下一个右侧节点指针 II 解法一样
     * <p>
     * 给定一个 完美二叉树 ，其所有叶子节点都在同一层，每个父节点都有两个子节点。二叉树定义如下：
     * <p>
     * struct Node {
     * int val;
     * Node *left;
     * Node *right;
     * Node *next;
     * }
     * 填充它的每个 next 指针，让这个指针指向其下一个右侧节点。如果找不到下一个右侧节点，则将 next 指针设置为 NULL。
     * <p>
     * 初始状态下，所有 next 指针都被设置为 NULL。
     *
     * @param root
     * @return
     */
    public Node116 leetCode116(Node116 root) {
        if (root == null) {
            return null;
        }

        /*
         * BFS
         */
        /*Queue<Node116> queue = new LinkedList<>();
        queue.offer(root);

        while (!queue.isEmpty()) {
            int len = queue.size();
            Node116 last = null;
            for (int i = 0; i < len; i++) {
                Node116 curNode = queue.poll();
                curNode.next = last;
                last = curNode;
                if (curNode.right != null) queue.offer(curNode.right);
                if (curNode.left != null) queue.offer(curNode.left);
            }
        }

        return root;*/

        /*
         * DFS
         */
        List<Node116> list = new ArrayList<>();
        dfs116(root, list, 0);
        return root;
    }

    private void dfs116(Node116 root, List<Node116> list, int curHeight) {
        if (list.size() == curHeight) {
            root.next = null;
            list.add(root);
        } else {
            root.next = list.get(curHeight);
            list.set(curHeight, root);
        }

        if (root.right != null) dfs116(root.right, list, curHeight + 1);
        if (root.left != null) dfs116(root.left, list, curHeight + 1);
    }

    /**
     * 104. 二叉树的最大深度
     * <p>
     * 给定一个二叉树，找出其最大深度。
     * <p>
     * 二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。
     * <p>
     * 说明: 叶子节点是指没有子节点的节点。
     *
     * @param root
     * @return
     */
    public int leetCode104(TreeNode root) {
        //BFS
        /*int result = 0;
        if (root == null) {
            return result;
        }

        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);

        while (!queue.isEmpty()) {
            result++;
            int len = queue.size();

            for (int i = 0; i < len; i++) {
                TreeNode curNode = queue.poll();
                if (curNode.left != null) queue.offer(curNode.left);
                if (curNode.right != null) queue.offer(curNode.right);
            }
        }

        return result;*/

        //DFS bySelf
        /*int result = 0;
        if (root == null) {
            return result;
        }
        List<TreeNode> list = new ArrayList<>();
        dfs104(root, list, 0);
        return list.size();*/

        //题解DFS
        if (root == null) return 0;
        return Math.max(leetCode104(root.left), leetCode104(root.right)) + 1;
    }

    private void dfs104(TreeNode root, List<TreeNode> list, int curHeight) {
        if (list.size() == curHeight) list.add(root);
        if (root.left != null) dfs104(root.left, list, curHeight + 1);
        if (root.right != null) dfs104(root.right, list, curHeight + 1);
    }

    /**
     * 111. 二叉树的最小深度
     * <p>
     * 给定一个二叉树，找出其最小深度。
     * <p>
     * 最小深度是从根节点到最近叶子节点的最短路径上的节点数量。
     * <p>
     * 说明：叶子节点是指没有子节点的节点。
     *
     * @param root
     * @return
     */
    public int leetCode111(TreeNode root) {
        int result = 0;
        if (root == null) {
            return result;
        }

        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);

        while (!queue.isEmpty()) {
            int len = queue.size();
            result++;

            for (int i = 0; i < len; i++) {
                TreeNode curNode = queue.poll();
                if (curNode.left == null && curNode.right == null) return result;
                if (curNode.left != null) queue.offer(curNode.left);
                if (curNode.right != null) queue.offer(curNode.right);
            }
        }

        return result;
    }

    /**
     * 226. 翻转二叉树
     * <p>
     * 给你一棵二叉树的根节点 root ，翻转这棵二叉树，并返回其根节点。
     *
     * @param root
     * @return
     */
    public TreeNode leetCode226(TreeNode root) {
        /*
         * 解题思路：前序遍历、后续遍历都可以 中序遍历比较麻烦 要注意翻转的问题
         */
        if (root == null) return null;

        leetCode226(root.left);
        leetCode226(root.right);

        TreeNode temp = root.left;
        root.left = root.right;
        root.right = temp;

        return root;
    }

    /**
     * 101. 对称二叉树
     * <p>
     * 给你一个二叉树的根节点 root ， 检查它是否轴对称。
     *
     * @param root
     * @return
     */
    public boolean leetCode101(TreeNode root) {
        /*
         * 解题思路：
         *  先确定base return：左右结点为空时的情况 只有同时为空时返回true
         *  不为空时 如果值不相等 直接返回false
         *          如果相等 同时比较左节点的左节点和右节点的右节点（外侧是否相同）
         *                  和左节点的右节点和右节点的左节点（内侧是否相同）
         * 最后一步有点类似于 两个后续遍历 左右中 和右左中 的比较
         */
        if (root == null) return false;
        return compare(root.left, root.right);
    }

    private boolean compare(TreeNode left, TreeNode right) {
        if (left == null && right != null) {
            return false;
        } else if (right == null && left != null) {
            return false;
        } else if (right == null) {
            return true;
        } else {
            if (right.val != left.val) return false;
            return compare(left.left, right.right) && compare(left.right, right.left);
        }
    }

    /**
     * 100. 相同的树
     * <p>
     * 给你两棵二叉树的根节点 p 和 q ，编写一个函数来检验这两棵树是否相同。
     * <p>
     * 如果两个树在结构上相同，并且节点具有相同的值，则认为它们是相同的。
     *
     * @param p
     * @param q
     * @return
     */
    public boolean leetCode100(TreeNode p, TreeNode q) {
        /*
         * 解析思路：同101
         */
        if (p == null && q != null) {
            return false;
        } else if (p != null && q == null) {
            return false;
        } else if (p == null && q == null) {
            return true;
        } else {
            if (p.val != q.val) return false;
            return leetCode100(p.left, q.left) && leetCode100(p.right, q.right);
        }
    }

    /**
     * 572. 另一棵树的子树
     * <p>
     * 给你两棵二叉树 root 和 subRoot 。检验 root 中是否包含和 subRoot 具有相同结构和节点值的子树。如果存在，返回 true ；否则，返回 false 。
     * <p>
     * 二叉树 tree 的一棵子树包括 tree 的某个节点和这个节点的所有后代节点。tree 也可以看做它自身的一棵子树。
     *
     * @param root
     * @param subRoot
     * @return
     */
    public boolean leetCode572(TreeNode root, TreeNode subRoot) {
        /*
         * 解题思路:
         *  暴力算法 前序遍历每个节点 调用leetCode100 判断两个树是否相等
         */
        if (leetCode100(root, subRoot)) return true;
        if (root.left != null && leetCode572(root.left, subRoot)) return true;
        return root.right != null && leetCode572(root.right, subRoot);
    }

    /**
     * 559. N 叉树的最大深度
     * <p>
     * 给定一个 N 叉树，找到其最大深度。
     * <p>
     * 最大深度是指从根节点到最远叶子节点的最长路径上的节点总数。
     * <p>
     * N 叉树输入按层序遍历序列化表示，每组子节点由空值分隔（请参见示例）。
     *
     * @param root
     * @return
     */
    public int leetCode559(Node root) {
        int result = 0;
        if (root == null) return result;

        Queue<Node> queue = new LinkedList<>();
        queue.offer(root);

        while (!queue.isEmpty()) {
            result++;
            int len = queue.size();

            for (int i = 0; i < len; i++) {
                Node curNode = queue.poll();
                for (Node node : curNode.children) {
                    if (node != null) queue.offer(node);
                }
            }
        }

        return result;
    }

    /**
     * 222. 完全二叉树的节点个数
     * <p>
     * 给你一棵 完全二叉树 的根节点 root ，求出该树的节点个数。
     * <p>
     * 完全二叉树 的定义如下：在完全二叉树中，除了最底层节点可能没填满外，其余每层节点数都达到最大值，并且最下面一层的节点都集中在该层最左边的若干位置。若最底层为第 h 层，则该层包含 1~ 2h 个节点。
     *
     * @param root
     * @return
     */
    public int leetCode222(TreeNode root) {
        /*
         * 解题思路：
         *  因为是一个完全二叉树 用一个节点的最左侧节点算作他的高度
         *  如果root的左节点高度等于右节点高度 说明左节点是满二叉树 那么继续按照次方式计算右节点 + 左节点所有结点数
         *  如果高度不相等 说明右节点是满二叉树 只不过右节点的高度比整个数的高度减一 此时继续按照方式极端左节点 + 右节点所有节点数
         */

        if (root == null) return 0;

        int leftHeight = getHeight(root.left);
        int rightHeight = getHeight(root.right);

        if (leftHeight == rightHeight) {
            int nextSum = leetCode222(root.right);
            int curSum = 1 << leftHeight;
            return nextSum + curSum;
        }

        int nextSum = leetCode222(root.left);
        int curSum = 1 << rightHeight;
        return nextSum + curSum;
    }

    private int getHeight(TreeNode root) {
        int result = 0;
        while (root != null) {
            root = root.left;
            result++;
        }
        return result;
    }

    /**
     * 110. 平衡二叉树
     * <p>
     * 给定一个二叉树，判断它是否是高度平衡的二叉树。
     * <p>
     * 本题中，一棵高度平衡二叉树定义为：
     * <p>
     * 一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过 1 。
     *
     * @param root
     * @return
     */
    public boolean leetCode110(TreeNode root) {
        /*
         * 1.解题思路：
         *  前序遍历 自顶向下 依次判断结点高度进行比较 但是会重复判断子节点高度 效率不高
         */
        /*if (root == null) return true;
        if (Math.abs(leetCode104(root.left) - leetCode104(root.right)) > 1) return false;
        return leetCode110(root.left) && leetCode110(root.right);*/

        /*
         * 解析思路2：
         *  2.后续遍历 自底向上 依次判断子节点高度进行比较 高度差大于1返回-1
         */
        return getBalanceHeight(root) != -1;
    }

    private int getBalanceHeight(TreeNode root) {
        if (root == null) return 0;

        int leftHeight = getBalanceHeight(root.left);
        if (leftHeight == -1) return -1;

        int rightHeight = getBalanceHeight(root.right);
        if (rightHeight == -1) return -1;

        if (Math.abs(leftHeight - rightHeight) > 1) return -1;
        return Math.max(leftHeight, rightHeight) + 1;
    }

    /**
     * 257. 二叉树的所有路径
     * <p>
     * 给你一个二叉树的根节点 root ，按 任意顺序 ，返回所有从根节点到叶子节点的路径。
     * <p>
     * 叶子节点 是指没有子节点的节点。
     *
     * @param root
     * @return
     */
    public List<String> leetCode257(TreeNode root) {
        /*
         * 解题思路：回溯算法
         */
        List<String> result = new ArrayList<>();
        List<Integer> path = new ArrayList<>();
        calcPath(root, result, path);
        return result;
    }

    private void calcPath(TreeNode root, List<String> result, List<Integer> path) {
        path.add(root.val);

        if (root.left == null && root.right == null) {
            result.add(path.stream().map(String::valueOf).collect(Collectors.joining("->")));
        }

        if (root.left != null) {
            calcPath(root.left, result, path);
            path.remove(path.size() - 1);
        }
        if (root.right != null) {
            calcPath(root.right, result, path);
            path.remove(path.size() - 1);
        }
    }

    /**
     * 404. 左叶子之和
     * <p>
     * 给定二叉树的根节点 root ，返回所有左叶子之和。
     *
     * @param root
     * @return
     */
    public int leetCode404(TreeNode root) {
        if (root == null) return 0;
        int left = leetCode404(root.left);
        int right = leetCode404(root.right);

        int sum = 0;
        if (root.left != null && root.left.left == null && root.left.right == null) {
            sum = root.left.val;
        }

        return left + right + sum;
    }

    /**
     * 513. 找树左下角的值
     * <p>
     * 给定一个二叉树的 根节点 root，请找出该二叉树的 最底层 最左边 节点的值。
     * <p>
     * 假设二叉树中至少有一个节点。
     *
     * @param root
     * @return
     */
    public int leetCode513(TreeNode root) {
        int result = 0;

        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);

        while (!queue.isEmpty()) {
            int len = queue.size();
            for (int i = 0; i < len; i++) {
                TreeNode curNode = queue.poll();
                if (i == 0) result = curNode.val;

                if (curNode.left != null) queue.offer(curNode.left);
                if (curNode.right != null) queue.offer(curNode.right);
            }
        }

        return result;
    }

    /**
     * 112. 路径总和
     * <p>
     * 给你二叉树的根节点 root 和一个表示目标和的整数 targetSum 。判断该树中是否存在 根节点到叶子节点 的路径，这条路径上所有节点值相加等于目标和 targetSum 。如果存在，返回 true ；否则，返回 false 。
     * <p>
     * 叶子节点 是指没有子节点的节点。
     *
     * @param root
     * @param targetSum
     * @return
     */
    public boolean leetCode112(TreeNode root, int targetSum) {
        if (root == null) return false;
        return getSum(root, 0, targetSum);
    }

    private boolean getSum(TreeNode root, int sum, int targetSum) {
        sum += root.val;
        //if (root.left == null && root.right == null && sum == targetSum) return true;
        // if(root.left != null && getSum(root.left, sum, targetSum)) return true;
        // if(root.right != null && getSum(root.right, sum, targetSum)) return true;
        // return false;

        //注释代码精简
        return (root.left == null && root.right == null && sum == targetSum)
                || (root.left != null && getSum(root.left, sum, targetSum))
                || (root.right != null && getSum(root.right, sum, targetSum));
    }

    /**
     * 113. 路径总和 II
     * <p>
     * 给你二叉树的根节点 root 和一个整数目标和 targetSum ，找出所有 从根节点到叶子节点 路径总和等于给定目标和的路径。
     * <p>
     * 叶子节点 是指没有子节点的节点。
     *
     * @param root
     * @param targetSum
     * @return
     */
    public List<List<Integer>> leetCode113(TreeNode root, int targetSum) {
        List<List<Integer>> result = new ArrayList<>();
        if (root == null) return result;
        List<Integer> path = new LinkedList<>();
        getPath(root, result, path, targetSum);
        return result;
    }

    private void getPath(TreeNode root, List<List<Integer>> result, List<Integer> path, int targetSum) {
        path.add(root.val);

        if (root.left == null && root.right == null
                //这个地方 可以优化的 修改每次target的值 判断最后一次 target是否和val相等
                && path.stream().mapToInt(Integer::intValue).sum() == targetSum) {
            //注意这里要new ArrayList
            result.add(new ArrayList<>(path));
            return;
        }

        if (root.left != null) {
            getPath(root.left, result, path, targetSum);
            path.remove(path.size() - 1);
        }
        if (root.right != null) {
            getPath(root.right, result, path, targetSum);
            path.remove(path.size() - 1);
        }
    }

    /**
     * 105. 从前序与中序遍历序列构造二叉树
     *
     * @param preorder
     * @param inorder
     * @return
     */
    public TreeNode leetCode105(int[] preorder, int[] inorder) {
        return buildTree108(preorder, 0, preorder.length - 1, inorder, 0, inorder.length - 1);//这个先写，你传入的参数肯定就是这几个
    }

    private TreeNode buildTree108(int[] preorder, int preStart, int preEnd, int[] inorder, int inStart, int inEnd) {
        //最后写递归出口 base case，很简单，就是两个数组之一越界就是出口（其实写1个也行，因为两个数组长度肯定相同的）
        if (preStart > preEnd || inStart > inEnd) {
            return null;
        }

        /*先序遍历框架-根、左、右*/
        //1.先构造根节点的值，做根节点
        //2.递归构造左子树
        //3.递归构造右子树

        //1.很明显根节点的值由先序遍历数组的第一个值确定
        int rootVal = preorder[preStart];
        TreeNode root = new TreeNode(rootVal);
        //2.递归构造左子树
        // root.left = bulid(preorder, ?, ?, inorder, ?, ?);//？需要我们画图去求的，也就是左右子树的索引
        // root.right = bulid(preorder, ?, ?, inorder, ?, ?);//？需要我们画图去求的，也就是左右子树的索引
        //首先通过rootVal在inorder中的索引（index），然后就能够知道inorder中左子树和右子树的边界
        //可以改进的，一开始用hashMap把inorder的值和索引存好，到时候直接查就行。
        int index = -1;
        for (int i = inStart; i <= inEnd; i++) {
            if (rootVal == inorder[i]) {
                index = i;
                break;
            }
        }
        //找到了index，确定inorder中左右子树的边界
        // root.left = bulid(preorder, ?, ?, inorder, inStart, index - 1);//确定inorder中左子树的边界
        // root.right = bulid(preorder, ?, ?, inorder, index + 1, inEnd);//确定inorder中右子树的边界
        //通过inorder中左子树节点的数目来确定preorder中左右子树的边界
        int nums_of_left_tree = index - inStart;
        // root.left = bulid(preorder, preStart + 1, preStart + nums_of_left_tree, inorder, ?, ?);//确定preorder中左子树的边界
        // root.right = bulid(preorder, preStart + nums_of_left_tree + 1, preEnd, inorder, ?, ?);//确定preorder中右子树的边界
        //最终确认好preorder和inorder中的左右子树边界
        root.left = buildTree108(preorder, preStart + 1, preStart + nums_of_left_tree, inorder, inStart, index - 1);
        root.right = buildTree108(preorder, preStart + nums_of_left_tree + 1, preEnd, inorder, index + 1, inEnd);
        return root;
    }

    /**
     * 654. 最大二叉树
     * <p>
     * 给定一个不重复的整数数组 nums 。 最大二叉树 可以用下面的算法从 nums 递归地构建:
     * <p>
     * 创建一个根节点，其值为 nums 中的最大值。
     * 递归地在最大值 左边 的 子数组前缀上 构建左子树。
     * 递归地在最大值 右边 的 子数组后缀上 构建右子树。
     * 返回 nums 构建的 最大二叉树 。
     *
     * @param nums
     * @return
     */
    public TreeNode leetCode654(int[] nums) {
        return buildTree108(0, nums.length - 1, nums);
    }

    private TreeNode buildTree108(int left, int right, int[] nums) {
        if (left > right) return null;

        int maxIndex = getMaxIndex(nums, left, right);
        TreeNode root = new TreeNode(nums[maxIndex]);

        root.left = buildTree108(left, maxIndex - 1, nums);
        root.right = buildTree108(maxIndex + 1, right, nums);

        return root;
    }

    private int getMaxIndex(int[] nums, int left, int right) {
        int max = Integer.MIN_VALUE;
        int index = -1;
        for (int i = left; i <= right; i++) {
            if (nums[i] > max) {
                max = nums[i];
                index = i;
            }
        }
        return index;
    }

    /**
     * 617. 合并二叉树
     *
     * @param root1
     * @param root2
     * @return
     */
    public TreeNode leetCode617(TreeNode root1, TreeNode root2) {
        if (root1 == null) {
            return root2;
        }
        if (root2 == null) {
            return root1;
        }

        TreeNode root = new TreeNode(root1.val + root2.val);
        root.left = leetCode617(root1.left, root2.left);
        root.right = leetCode617(root1.right, root2.right);

        return root;
    }

    /**
     * 700. 二叉搜索树中的搜索
     * <p>
     * 给定二叉搜索树（BST）的根节点 root 和一个整数值 val。
     * <p>
     * 你需要在 BST 中找到节点值等于 val 的节点。 返回以该节点为根的子树。 如果节点不存在，则返回 null 。
     *
     * @param root
     * @param val
     * @return
     */
    public TreeNode leetCode700(TreeNode root, int val) {
        if (root == null) return null;

        if (root.val == val) return root;

        return leetCode700(root.val > val ? root.left : root.right, val);
    }

    /**
     * 98. 验证二叉搜索树
     * <p>
     * 给你一个二叉树的根节点 root ，判断其是否是一个有效的二叉搜索树。
     * <p>
     * 有效 二叉搜索树定义如下：
     * <p>
     * 节点的左子树只包含 小于 当前节点的数。
     * 节点的右子树只包含 大于 当前节点的数。
     * 所有左子树和右子树自身必须也是二叉搜索树。
     *
     * @param root
     * @return
     */
    TreeNode max98;

    public boolean leetCode98(TreeNode root) {
        if (root == null) return true;

        if (!leetCode98(root.left)) return false;

        if (max98 != null && max98.val >= root.val) return false;
        max98 = root;

        return leetCode98(root.right);
    }


    /**
     * 530. 二叉搜索树的最小绝对差
     * <p>
     * 给你一个二叉搜索树的根节点 root ，返回 树中任意两不同节点值之间的最小差值 。
     * <p>
     * 差值是一个正数，其数值等于两值之差的绝对值。
     *
     * @param root
     * @return
     */
    Integer min = Integer.MAX_VALUE;
    TreeNode last;

    public int leetCode530(TreeNode root) {
        traverse530(root);
        return min;
    }

    private void traverse530(TreeNode root) {
        if (root == null) return;

        traverse530(root.left);

        if (last != null) {
            min = Math.min(root.val - last.val, min);
        }
        last = root;

        traverse530(root.right);
    }

    /**
     * 501. 二叉搜索树中的众数
     * <p>
     * 给你一个含重复值的二叉搜索树（BST）的根节点 root ，找出并返回 BST 中的所有 众数（即，出现频率最高的元素）。
     * <p>
     * 如果树中有不止一个众数，可以按 任意顺序 返回。
     *
     * @param root
     * @return
     */
    List<Integer> list = new ArrayList<>();
    int count = 0;
    int max = 0;
    TreeNode pre;

    public int[] leetCode501(TreeNode root) {
        traverse501(root);
        return list.stream().mapToInt(Integer::intValue).toArray();
    }

    private void traverse501(TreeNode root) {
        if (root == null) return;

        traverse501(root.left);

        if (pre == null || root.val != pre.val) {
            count = 1;
        } else {
            count++;
        }

        if (count > max) {
            max = count;
            list.clear();
            list.add(root.val);
        } else if (count == max) {
            list.add(root.val);
        }

        pre = root;

        traverse501(root.right);
    }

    /**
     * 236. 二叉树的最近公共祖先
     * <p>
     * 给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。
     * <p>
     * 百度百科中最近公共祖先的定义为：“对于有根树 T 的两个节点 p、q，最近公共祖先表示为一个节点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”
     *
     * @param root
     * @param p
     * @param q
     * @return
     */
    public TreeNode leetCode236(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null) return null;

        //找到p或q结点就返回
        if (root == p || root == q) return root;

        //分别找左右子树 是否存在p、q结点
        TreeNode leftFind = leetCode236(root.left, p, q);
        TreeNode rightFind = leetCode236(root.right, p, q);

        //p、q分别在左右子树中 返回当前结点root 因为是后续遍历 所以是最深的
        if (leftFind != null && rightFind != null) return root;

        //p、q在同一侧 即p与q是包含的关系 则返回对应存在的结点
        return leftFind != null ? leftFind : rightFind;
    }

    /**
     * 235. 二叉搜索树的最近公共祖先
     * <p>
     * 给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。
     * <p>
     * 百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”
     *
     * @param root
     * @param p
     * @param q
     * @return
     */
    public TreeNode leetCode235(TreeNode root, TreeNode p, TreeNode q) {
        int min = Math.min(p.val, q.val);
        int max = Math.max(p.val, q.val);
        return find235(root, min, max);
    }

    private TreeNode find235(TreeNode root, int min, int max) {
        if (root == null) return null;
        //如果 当前值比最小值还小 查找右子树
        if (root.val < min) return find235(root.right, min, max);
        //如果 当前值比最大值还大 查找左子树
        if (root.val > max) return find235(root.left, min, max);
        return root;
    }

    /**
     * 701. 二叉搜索树中的插入操作
     * <p>
     * 给定二叉搜索树（BST）的根节点 root 和要插入树中的值 value ，将值插入二叉搜索树。 返回插入后二叉搜索树的根节点。 输入数据 保证 ，新值和原始二叉搜索树中的任意节点值都不同。
     * <p>
     * 注意，可能存在多种有效的插入方式，只要树在插入后仍保持为二叉搜索树即可。 你可以返回 任意有效的结果 。
     *
     * @param root
     * @param val
     * @return
     */
    public TreeNode leetCode701(TreeNode root, int val) {
        /*
         * 解题思路：
         *  自顶向下 判断是插在左子树还是右子树 直到找到可以插入的地方
         */
        if (root == null) return new TreeNode(val);

        TreeNode node = root;
        while (node != null) {
            if (val < node.val) {
                if (node.left == null) {
                    node.left = new TreeNode(val);
                    return root;
                }
                node = node.left;
            } else {
                if (node.right == null) {
                    node.right = new TreeNode(val);
                    return root;
                }
                node = node.right;
            }
        }
        return root;
    }

    /**
     * 450. 删除二叉搜索树中的节点
     * <p>
     * 给定一个二叉搜索树的根节点 root 和一个值 key，删除二叉搜索树中的 key 对应的节点，并保证二叉搜索树的性质不变。返回二叉搜索树（有可能被更新）的根节点的引用。
     * <p>
     * 一般来说，删除节点可分为两个步骤：
     * <p>
     * 首先找到需要删除的节点；
     * 如果找到了，删除它。
     *
     * @param root
     * @param key
     * @return
     */
    public TreeNode leetCode450(TreeNode root, int key) {
        /*
         * 解题思路：
         *  自顶向下 key小于root.val 遍历左子树 相反遍历右子树  递归返回值为被删除的元素的新的根节点（父节点）
         *  当找到删除节点时 分为下面四中请款：
         *      1.左空、右空 即叶子节点 直接null为新的根节点
         *      2.左空、右有 返回右节点为新的根节点
         *      3.有空、左有 返回左节点为新的根节点
         *      4.左右都有 将左节点挂在右节点的最左侧 返回右节点为新的根节点
         */
        if (root == null) return null;
        //找到了
        if (root.val == key) {
            //情况1和情况2
            if (root.left == null) return root.right;
            //情况3
            if (root.right == null) return root.left;
            //情况4
            //找到右子树的最左节点
            TreeNode rightMin = root.right;
            while (rightMin.left != null) {
                rightMin = rightMin.left;
            }
            //将左子树挂在最左节点的左侧
            rightMin.left = root.left;
            return root.right;
        }

        if (root.val > key) root.left = leetCode450(root.left, key);
        if (root.val < key) root.right = leetCode450(root.right, key);

        return root;
    }

    /**
     * 669. 修剪二叉搜索树
     * <p>
     * 给你二叉搜索树的根节点 root ，同时给定最小边界low 和最大边界 high。通过修剪二叉搜索树，使得所有节点的值在[low, high]中。修剪树 不应该 改变保留在树中的元素的相对结构 (即，如果没有被移除，原有的父代子代关系都应当保留)。 可以证明，存在 唯一的答案 。
     * <p>
     * 所以结果应当返回修剪好的二叉搜索树的新的根节点。注意，根节点可能会根据给定的边界发生改变。
     *
     * @param root
     * @param low
     * @param high
     * @return
     */
    public TreeNode leetCode669(TreeNode root, int low, int high) {
        if (root == null) return null;

        //小于最小边界 说明左子树都是无效数据 返回右子树处理结果 同理大于最大边界
        if (root.val < low) return leetCode669(root.right, low, high);
        if (root.val > high) return leetCode669(root.left, low, high);

        //走到这 说明都是在区间内的 根节点的左子树等于左子树处理结果 右子树同理
        root.left = leetCode669(root.left, low, high);
        root.right = leetCode669(root.right, low, high);

        return root;
    }

    /**
     * 108. 将有序数组转换为二叉搜索树
     * 给你一个整数数组 nums ，其中元素已经按 升序 排列，请你将其转换为一棵 高度平衡 二叉搜索树。
     * <p>
     * 高度平衡 二叉树是一棵满足「每个节点的左右两个子树的高度差的绝对值不超过 1 」的二叉树。
     *
     * @param nums
     * @return
     */
    public TreeNode leetCode108(int[] nums) {
        return buildTree108(nums, 0, nums.length - 1);
    }

    private TreeNode buildTree108(int[] nums, int left, int right) {
        if (right < left) return null;
        //注意 大int越界问题
        int mid = left + ((right - left) / 2);
        return new TreeNode(nums[mid], buildTree108(nums, left, mid - 1), buildTree108(nums, mid + 1, right));
    }

    /**
     * 538. 把二叉搜索树转换为累加树  与leetCode1038相同
     * <p>
     * 给出二叉 搜索 树的根节点，该树的节点值各不相同，请你将其转换为累加树（Greater Sum Tree），使每个节点 node 的新值等于原树中大于或等于 node.val 的值之和。
     *
     * @param root
     * @return
     */
    int sum538 = 0;

    public TreeNode leetCode538(TreeNode root) {
        /*
         * 解题思路：右->中->左 逆向中序遍历
         */
        traverse538(root);
        return root;
    }

    private void traverse538(TreeNode root) {
        if (root == null) return;

        traverse538(root.right);

        root.val += sum538;
        sum538 = root.val;

        traverse538(root.left);
    }
}
